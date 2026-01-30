import pickle
from contextlib import nullcontext
from pathlib import Path

import faiss
import numpy as np
from langchain_community.retrievers import BM25Retriever

from state.config import (
    EMBEDDING_DIMENSIONS,
    TOKENS_PER_CHUNK,
    WORDS_PER_CHUNK_OVERLAP,
    TOP_K_TEXT_BM25,
)
from ingestion.cleaners import remove_junk_sections, remove_junk_lines
from ingestion.chunkers import chunk_text2
from ingestion.embeddings import estimate_embedding_cost
from ingestion.faiss_store import build_faiss_from_embeddings, load_faiss_from_disk
from ingestion.graphrag import run_graphrag_cli
from ingestion.loaders import extract_text_from_pdf


class KnowledgeBaseError(Exception):
    """Base error for knowledge-base operations."""


class KnowledgeBaseBuildError(KnowledgeBaseError):
    """Raised when building a knowledge base fails."""


class KnowledgeBaseLoadError(KnowledgeBaseError):
    """Raised when loading a knowledge base fails."""


class KnowledgeBaseAppendError(KnowledgeBaseError):
    """Raised when appending to a knowledge base fails."""


class GraphRAGWorkspaceError(KnowledgeBaseError):
    """Raised when GraphRAG workspace is invalid or missing."""


def _get_cb(callbacks, name):
    if callbacks is None:
        return None
    return getattr(callbacks, name, None)


def _call(cb, *args, **kwargs):
    if cb:
        cb(*args, **kwargs)


def _spinner(callbacks, message):
    cb = _get_cb(callbacks, "spinner")
    if cb:
        return cb(message)
    return nullcontext()


class KnowledgeBaseBuilder:
    """Build, load, and register a knowledge base."""

    def __init__(self, client, embeddings):
        self.client = client
        self.embeddings = embeddings
        import streamlit as st

        self._st = st
        self.enc = st.session_state.enc

    def build(self, pdf_dir, index_path, meta_path, graphrag_dir, model, callbacks=None):
        dim = EMBEDDING_DIMENSIONS[model]

        texts, chunks = {}, {}
        total_chunks = 0
        total_tokens = 0
        total_cost = 0.0
        embeddings, metadata = [], []

        _call(_get_cb(callbacks, "info"), "🔍 Processing PDFs...")
        pdf_files = list(Path(pdf_dir).glob("*.pdf"))

        for i, pdf in enumerate(pdf_files):
            try:
                text, cks = self._process_pdf(pdf)
                texts[pdf.name] = text
                chunks[pdf.name] = cks

                token_count = sum(len(self.enc.encode(c)) for c in cks)
                total_cost += estimate_embedding_cost(token_count, model)
                total_chunks += len(cks)
                total_tokens += token_count
            except Exception as exc:
                _call(
                    _get_cb(callbacks, "warning"),
                    f"⚠️ Failed on {pdf.name}: {exc}",
                )
            _call(
                _get_cb(callbacks, "progress"),
                "processing_pdfs",
                i + 1,
                max(len(pdf_files), 1),
            )

        if total_chunks == 0:
            raise KnowledgeBaseBuildError("No chunks produced; aborting.")

        _call(
            _get_cb(callbacks, "info"),
            (
                f"📦 Total chunks: {total_chunks:,}, "
                f"Total tokens: {total_tokens:,}, "
                f"Est. cost: ${total_cost:.4f}"
            ),
        )

        _call(_get_cb(callbacks, "info"), "📌 Embedding and indexing...")
        chunk_count = 0

        for filename, cks in chunks.items():
            for j, chunk in enumerate(cks):
                try:
                    emb = self.client.embeddings.create(
                        input=chunk, model=model
                    ).data[0].embedding
                    embeddings.append(emb)
                    metadata.append(
                        {
                            "source": filename,
                            "chunk_id": j,
                            "text": chunk,
                            "embedding_model": model,
                        }
                    )
                except Exception as exc:
                    _call(
                        _get_cb(callbacks, "warning"),
                        f"⚠️ Embedding failed: {filename}, chunk {j} → {exc}",
                    )
                chunk_count += 1
                _call(
                    _get_cb(callbacks, "progress"),
                    "embedding",
                    chunk_count,
                    total_chunks,
                )

        index, db, docs = build_faiss_from_embeddings(
            embeddings, metadata, self.embeddings, dim
        )

        Path(index_path).parent.mkdir(parents=True, exist_ok=True)
        Path(meta_path).parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(index, index_path)
        with open(meta_path, "wb") as f:
            pickle.dump(metadata, f)

        self._run_graphrag(texts, graphrag_dir)
        self._register(index, metadata, db, docs, graphrag_dir)

    def load(self, index_path, meta_path, graphrag_dir):
        try:
            index = faiss.read_index(index_path)
            with open(meta_path, "rb") as f:
                metadata = pickle.load(f)

            model = metadata[0]["embedding_model"]
            dim = EMBEDDING_DIMENSIONS[model]

            db, docs = load_faiss_from_disk(index, metadata, self.embeddings)
            self._register(index, metadata, db, docs, graphrag_dir)

        except Exception as exc:
            raise KnowledgeBaseLoadError("Failed to load knowledge base.") from exc

    def append(self, index_path, meta_path, append_folder, graphrag_dir, callbacks=None):
        """Append new PDFs to an existing knowledge base."""
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

        existing_model = metadata_existing[0].get(
            "embedding_model", self._st.session_state.get("embedding_model")
        )
        self._st.session_state.embedding_model = existing_model
        dim = EMBEDDING_DIMENSIONS[existing_model]

        _call(_get_cb(callbacks, "info"), "🔍 Processing NEW PDFs to append...")
        new_chunks_by_file = {}
        all_texts = {}
        total_new_chunks = 0
        total_new_tokens = 0
        total_new_cost = 0.0

        for i, pdf_path in enumerate(pdf_files):
            try:
                text, chunks = self._process_pdf(pdf_path)
                all_texts[pdf_path.name] = text
                new_chunks_by_file[pdf_path.name] = chunks
                token_count = sum(len(self.enc.encode(c)) for c in chunks)
                total_new_cost += estimate_embedding_cost(token_count, existing_model)
                total_new_chunks += len(chunks)
                total_new_tokens += token_count
            except Exception as exc:
                _call(
                    _get_cb(callbacks, "warning"),
                    f"⚠️ Failed on {pdf_path.name}: {exc}",
                )
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

        _call(
            _get_cb(callbacks, "info"),
            (
                f"📦 New chunks: {total_new_chunks:,}, "
                f"New tokens: {total_new_tokens:,}, "
                f"Est. cost: ${total_new_cost:.4f}"
            ),
        )

        _call(_get_cb(callbacks, "info"), "📌 Embedding and appending to FAISS...")
        new_embeddings = []
        new_metadata = []
        count = 0

        for filename, chunks in new_chunks_by_file.items():
            for j, chunk in enumerate(chunks):
                try:
                    resp = self.client.embeddings.create(
                        input=chunk, model=existing_model
                    )
                    new_embeddings.append(resp.data[0].embedding)
                    new_metadata.append(
                        {
                            "source": filename,
                            "chunk_id": j,
                            "text": chunk,
                            "embedding_model": existing_model,
                        }
                    )
                except Exception as exc:
                    _call(
                        _get_cb(callbacks, "warning"),
                        f"⚠️ Embedding failed: {filename}, chunk {j} → {exc}",
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

        db, docs = load_faiss_from_disk(index, updated_metadata, self.embeddings)
        self._register(index, updated_metadata, db, docs, graphrag_dir)

        with _spinner(callbacks, "🕸️ Updating Knowledge-Graph with new documents..."):
            try:
                self._append_graphrag(all_texts, graphrag_dir)
                _call(
                    _get_cb(callbacks, "success"),
                    "✅ Knowledge-Graph updated successfully!",
                )
            except Exception as exc:
                _call(
                    _get_cb(callbacks, "warning"),
                    f"⚠️ Knowledge-Graph update failed: {exc}",
                )

    def _process_pdf(self, path):
        text = extract_text_from_pdf(path)
        text = remove_junk_lines(remove_junk_sections(text))
        chunks = chunk_text2(
            text,
            max_tokens=TOKENS_PER_CHUNK,
            tokenizer=self.enc,
            overlap=WORDS_PER_CHUNK_OVERLAP,
        )
        return text, chunks

    def _run_graphrag(self, texts, root):
        root = Path(root)
        (root / "input").mkdir(parents=True, exist_ok=True)
        (root / "output").mkdir(parents=True, exist_ok=True)

        for name, text in texts.items():
            (root / "input" / f"{Path(name).stem}.txt").write_text(text)

        run_graphrag_cli(
            root, root / "input", root / "output", self._st.session_state.api_key
        )

    def _append_graphrag(self, texts, root):
        """Append new docs to an existing GraphRAG workspace."""
        root = Path(root)
        input_dir = root / "input"
        output_dir = root / "output"
        if not input_dir.is_dir() or not output_dir.is_dir():
            raise GraphRAGWorkspaceError(
                "GraphRAG workspace is missing input/output folders."
            )

        for name, text in texts.items():
            (input_dir / f"{Path(name).stem}.txt").write_text(text)

        run_graphrag_cli(
            root, input_dir, output_dir, self._st.session_state.api_key
        )

    def _register(self, index, metadata, db, docs, graphrag_dir):
        self._st.session_state.update(
            index=index,
            metadata=metadata,
            db=db,
            all_documents=docs,
            bm25=BM25Retriever.from_documents(docs, k=TOP_K_TEXT_BM25),
            embedding_model=metadata[0]["embedding_model"],
            dimension=EMBEDDING_DIMENSIONS[metadata[0]["embedding_model"]],
            graphrag=graphrag_dir,
        )
