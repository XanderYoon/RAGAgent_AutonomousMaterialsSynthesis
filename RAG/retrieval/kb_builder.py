import pickle
from pathlib import Path

import faiss
import streamlit as st
from langchain_community.retrievers import BM25Retriever

from config import (
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


class KnowledgeBaseBuilder:
    """Build, load, and register a knowledge base."""

    def __init__(self, client, embeddings):
        self.client = client
        self.embeddings = embeddings
        self.enc = st.session_state.enc

    def build(self, pdf_dir, index_path, meta_path, graphrag_dir, model):
        dim = EMBEDDING_DIMENSIONS[model]

        texts, chunks = {}, {}
        embeddings, metadata = [], []

        for pdf in Path(pdf_dir).glob("*.pdf"):
            text, cks = self._process_pdf(pdf)
            texts[pdf.name] = text
            chunks[pdf.name] = cks

            token_count = sum(len(self.enc.encode(c)) for c in cks)
            estimate_embedding_cost(token_count, model)

            for i, chunk in enumerate(cks):
                emb = self.client.embeddings.create(
                    input=chunk, model=model
                ).data[0].embedding
                embeddings.append(emb)
                metadata.append(
                    {
                        "source": pdf.name,
                        "chunk_id": i,
                        "text": chunk,
                        "embedding_model": model,
                    }
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
        index = faiss.read_index(index_path)
        with open(meta_path, "rb") as f:
            metadata = pickle.load(f)

        model = metadata[0]["embedding_model"]
        dim = EMBEDDING_DIMENSIONS[model]

        db, docs = load_faiss_from_disk(index, metadata, self.embeddings)
        self._register(index, metadata, db, docs, graphrag_dir)

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
            root, root / "input", root / "output", st.session_state.api_key
        )

    def _register(self, index, metadata, db, docs, graphrag_dir):
        st.session_state.update(
            index=index,
            metadata=metadata,
            db=db,
            all_documents=docs,
            bm25=BM25Retriever.from_documents(docs, k=TOP_K_TEXT_BM25),
            embedding_model=metadata[0]["embedding_model"],
            dimension=EMBEDDING_DIMENSIONS[metadata[0]["embedding_model"]],
            graphrag=graphrag_dir,
        )
