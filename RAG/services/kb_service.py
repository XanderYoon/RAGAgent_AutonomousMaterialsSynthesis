import os
import pickle
from pathlib import Path

import faiss
import numpy as np

from ingestion.chunkers import chunk_text2
from ingestion.cleaners import remove_junk_lines, remove_junk_sections
from ingestion.embeddings import estimate_embedding_cost
from ingestion.faiss_store import build_faiss_from_embeddings
from ingestion.graphrag import run_graphrag_cli
from ingestion.loaders import extract_text_from_pdf
from state.config import EMBEDDING_DIMENSIONS, TOKENS_PER_CHUNK, WORDS_PER_CHUNK_OVERLAP
from state.schemas import ChunkMetadata
from RAG.retrieval.kb_builder import (
    GraphRAGWorkspaceError,
    KnowledgeBaseAppendError,
    KnowledgeBaseBuildError,
    KnowledgeBaseLoadError,
)


def _get_cb(callbacks, name):
    if callbacks is None:
        return None
    return getattr(callbacks, name, None)


def _call(cb, *args, **kwargs):
    if cb:
        cb(*args, **kwargs)


def _resolve_api_key(api_key: str | None) -> str | None:
    return api_key or os.environ.get("OPENAI_API_KEY")


def _process_pdf(path: Path, enc):
    text = extract_text_from_pdf(path)
    text = remove_junk_lines(remove_junk_sections(text))
    chunks = chunk_text2(
        text,
        max_tokens=TOKENS_PER_CHUNK,
        tokenizer=enc,
        overlap=WORDS_PER_CHUNK_OVERLAP,
    )
    return text, chunks


def _run_graphrag(texts, graphrag_dir: str, api_key: str | None, warnings):
    if not graphrag_dir:
        return

    key = _resolve_api_key(api_key)
    if not key:
        warnings.append(
            "Skipping GraphRAG indexing: no API key provided or found in env."
        )
        return

    root = Path(graphrag_dir)
    (root / "input").mkdir(parents=True, exist_ok=True)
    (root / "output").mkdir(parents=True, exist_ok=True)

    for name, text in texts.items():
        (root / "input" / f"{Path(name).stem}.txt").write_text(text)

    ok = run_graphrag_cli(root, root / "input", root / "output", key)
    if not ok:
        warnings.append("GraphRAG indexing failed; see logs for details.")


def _append_graphrag(texts, graphrag_dir: str, api_key: str | None):
    if not graphrag_dir:
        return

    key = _resolve_api_key(api_key)
    if not key:
        raise GraphRAGWorkspaceError(
            "GraphRAG update requested but no API key was provided."
        )

    root = Path(graphrag_dir)
    input_dir = root / "input"
    output_dir = root / "output"
    if not input_dir.is_dir() or not output_dir.is_dir():
        raise GraphRAGWorkspaceError(
            "GraphRAG workspace is missing input/output folders."
        )

    for name, text in texts.items():
        (input_dir / f"{Path(name).stem}.txt").write_text(text)

    run_graphrag_cli(root, input_dir, output_dir, key)


def build_kb(
    *,
    client,
    embeddings,
    enc,
    pdf_dir: str,
    index_path: str,
    meta_path: str,
    graphrag_dir: str | None,
    embedding_model: str,
    run_graphrag: bool = True,
    api_key: str | None = None,
    callbacks=None,
):
    if embedding_model not in EMBEDDING_DIMENSIONS:
        raise KnowledgeBaseBuildError(
            f"Unsupported embedding model: {embedding_model}"
        )

    dim = EMBEDDING_DIMENSIONS[embedding_model]
    pdf_files = list(Path(pdf_dir).glob("*.pdf"))
    if not pdf_files:
        raise KnowledgeBaseBuildError("No PDFs found in the provided directory.")

    texts = {}
    chunks = {}
    warnings = []
    total_chunks = 0
    total_tokens = 0
    total_cost = 0.0

    _call(_get_cb(callbacks, "info"), "Processing PDFs...")
    for i, pdf in enumerate(pdf_files):
        try:
            text, cks = _process_pdf(pdf, enc)
            texts[pdf.name] = text
            chunks[pdf.name] = cks

            token_count = sum(len(enc.encode(c)) for c in cks)
            total_cost += estimate_embedding_cost(token_count, embedding_model)
            total_chunks += len(cks)
            total_tokens += token_count
        except Exception as exc:
            warnings.append(f"Failed on {pdf.name}: {exc}")
        _call(
            _get_cb(callbacks, "progress"),
            "processing_pdfs",
            i + 1,
            max(len(pdf_files), 1),
        )

    if total_chunks == 0:
        raise KnowledgeBaseBuildError("No chunks produced; aborting.")

    embeddings_list = []
    metadata = []
    _call(_get_cb(callbacks, "info"), "Embedding and indexing...")

    chunk_count = 0
    for filename, cks in chunks.items():
        for j, chunk in enumerate(cks):
            try:
                emb = client.embeddings.create(
                    input=chunk, model=embedding_model
                ).data[0].embedding
                embeddings_list.append(emb)
                meta = ChunkMetadata(
                    source=filename,
                    chunk_id=j,
                    text=chunk,
                    embedding_model=embedding_model,
                )
                metadata.append(meta.model_dump())
            except Exception as exc:
                warnings.append(f"Embedding failed: {filename}, chunk {j} -> {exc}")
            chunk_count += 1
            _call(
                _get_cb(callbacks, "progress"),
                "embedding",
                chunk_count,
                total_chunks,
            )

    if not embeddings_list:
        raise KnowledgeBaseBuildError("No embeddings computed; aborting.")

    index, _, _ = build_faiss_from_embeddings(
        embeddings_list, metadata, embeddings, dim
    )

    Path(index_path).parent.mkdir(parents=True, exist_ok=True)
    Path(meta_path).parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, index_path)
    with open(meta_path, "wb") as f:
        pickle.dump(metadata, f)

    if run_graphrag and graphrag_dir:
        _run_graphrag(texts, graphrag_dir, api_key, warnings)

    return {
        "total_chunks": total_chunks,
        "total_tokens": total_tokens,
        "estimated_cost": total_cost,
        "warnings": warnings,
    }


def load_kb(*, index_path: str, meta_path: str, graphrag_dir: str | None):
    try:
        index = faiss.read_index(index_path)
        _ = index  # keep the local name used for basic validation
        with open(meta_path, "rb") as f:
            metadata = pickle.load(f)
        if not metadata:
            raise KnowledgeBaseLoadError("Metadata file is empty.")
        model = metadata[0]["embedding_model"]
        dim = EMBEDDING_DIMENSIONS[model]
    except KnowledgeBaseLoadError:
        raise
    except Exception as exc:
        raise KnowledgeBaseLoadError("Failed to load knowledge base.") from exc

    return {
        "status": "ok",
        "embedding_model": model,
        "dimension": dim,
        "chunk_count": len(metadata),
        "graphrag_dir": graphrag_dir or "",
    }


def append_kb(
    *,
    client,
    enc,
    index_path: str,
    meta_path: str,
    append_folder: str,
    graphrag_dir: str | None,
    run_graphrag: bool = True,
    api_key: str | None = None,
    callbacks=None,
):
    index_path = Path(index_path)
    meta_path = Path(meta_path)
    append_folder = Path(append_folder)

    if not index_path.is_file():
        raise KnowledgeBaseAppendError("Existing FAISS index file not found.")
    if not meta_path.is_file():
        raise KnowledgeBaseAppendError("Existing metadata file not found.")
    if not append_folder.is_dir():
        raise KnowledgeBaseAppendError("Append folder does not exist.")

    pdf_files = list(append_folder.glob("*.pdf"))
    if not pdf_files:
        raise KnowledgeBaseAppendError("No PDFs found in the append folder.")

    try:
        index = faiss.read_index(str(index_path))
        with open(meta_path, "rb") as f:
            metadata_existing = pickle.load(f)
    except Exception as exc:
        raise KnowledgeBaseAppendError(
            "Failed to load existing knowledge base."
        ) from exc

    existing_model = metadata_existing[0].get("embedding_model")
    if existing_model not in EMBEDDING_DIMENSIONS:
        raise KnowledgeBaseAppendError("Unknown embedding model in metadata.")
    dim = EMBEDDING_DIMENSIONS[existing_model]

    warnings = []
    new_chunks_by_file = {}
    all_texts = {}
    total_new_chunks = 0
    total_new_tokens = 0
    total_new_cost = 0.0

    _call(_get_cb(callbacks, "info"), "Processing NEW PDFs to append...")
    for i, pdf_path in enumerate(pdf_files):
        try:
            text, chunks = _process_pdf(pdf_path, enc)
            all_texts[pdf_path.name] = text
            new_chunks_by_file[pdf_path.name] = chunks
            token_count = sum(len(enc.encode(c)) for c in chunks)
            total_new_cost += estimate_embedding_cost(token_count, existing_model)
            total_new_chunks += len(chunks)
            total_new_tokens += token_count
        except Exception as exc:
            warnings.append(f"Failed on {pdf_path.name}: {exc}")
        _call(
            _get_cb(callbacks, "progress"),
            "processing_new_pdfs",
            i + 1,
            len(pdf_files),
        )

    if total_new_chunks == 0:
        raise KnowledgeBaseAppendError(
            "No new chunks were produced from the append PDFs."
        )

    _call(_get_cb(callbacks, "info"), "Embedding and appending to FAISS...")
    new_embeddings = []
    new_metadata = []
    count = 0

    for filename, chunks in new_chunks_by_file.items():
        for j, chunk in enumerate(chunks):
            try:
                resp = client.embeddings.create(
                    input=chunk, model=existing_model
                )
                new_embeddings.append(resp.data[0].embedding)
                meta = ChunkMetadata(
                    source=filename,
                    chunk_id=j,
                    text=chunk,
                    embedding_model=existing_model,
                )
                new_metadata.append(meta.model_dump())
            except Exception as exc:
                warnings.append(
                    f"Embedding failed: {filename}, chunk {j} -> {exc}"
                )
            count += 1
            _call(
                _get_cb(callbacks, "progress"),
                "embedding_append",
                count,
                total_new_chunks,
            )

    if not new_embeddings:
        raise KnowledgeBaseAppendError("No new embeddings computed; aborting.")

    new_mat = np.array(new_embeddings, dtype="float32")
    if new_mat.shape[1] != dim:
        raise KnowledgeBaseAppendError("Embedding dimension mismatch. Aborting.")

    try:
        index.add(new_mat)
    except Exception as exc:
        raise KnowledgeBaseAppendError(
            "Failed to append vectors to FAISS."
        ) from exc

    updated_metadata = metadata_existing + new_metadata
    index_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    with open(meta_path, "wb") as f:
        pickle.dump(updated_metadata, f)

    if run_graphrag and graphrag_dir:
        try:
            _append_graphrag(all_texts, graphrag_dir, api_key)
        except GraphRAGWorkspaceError as exc:
            warnings.append(str(exc))

    return {
        "new_chunks": total_new_chunks,
        "new_tokens": total_new_tokens,
        "estimated_cost": total_new_cost,
        "warnings": warnings,
    }
