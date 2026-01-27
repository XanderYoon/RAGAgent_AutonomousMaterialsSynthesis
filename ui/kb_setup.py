# ui/kb_setup.py
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
from ingestion.loaders import extract_text_from_pdf
from ingestion.embeddings import estimate_embedding_cost
from ingestion.faiss_store import (
    build_faiss_from_embeddings,
    load_faiss_from_disk,
)
from ingestion.graphrag import run_graphrag_cli

# -------------------------
# Knowledge Base Logic
# -------------------------

class KnowledgeBaseBuilder:
    """Build, load, and register a knowledge base."""

    def __init__(self, client, embeddings):
        self.client = client
        self.embeddings = embeddings
        self.enc = st.session_state.enc

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


# -------------------------
# UI
# -------------------------

def kb_setup(client, embeddings):
    st.header("📂 Knowledge-Base Setup")
    has_valid_key = st.session_state.get("api_verified", False)
    if not has_valid_key:
        st.info("ℹ️ Please set a valid API key.")

    kb = KnowledgeBaseBuilder(client, embeddings) if has_valid_key else None

    mode = st.radio(
        "Choose setup method:",
        ("📚 Build", "📤 Load", "➕ Append"),
    )

    if mode == "📚 Build":
        model = st.selectbox(
            "Embedding model",
            ["text-embedding-3-small", "text-embedding-3-large"],
        )
        pdf_dir = st.text_input("PDF folder", "inputs", disabled = not has_valid_key)
        index_path = st.text_input("Index path", "outputs/index.index", disabled = not has_valid_key)
        meta_path = st.text_input("Metadata path", "outputs/meta.pkl", disabled = not has_valid_key)
        graphrag_dir = st.text_input("GraphRAG dir", "outputs/graphrag", disabled = not has_valid_key)

        if st.button("Build",disabled = not has_valid_key):
            kb.build(pdf_dir, index_path, meta_path, graphrag_dir, model)
            st.success("✅ Knowledge-Base built")

    elif mode == "📤 Load":
        index_path = st.text_input("Index path", disabled = not has_valid_key)
        meta_path = st.text_input("Metadata path", disabled = not has_valid_key)
        graphrag_dir = st.text_input("GraphRAG dir", disabled = not has_valid_key)

        if st.button("Load", disabled = not has_valid_key):
            kb.load(index_path, meta_path, graphrag_dir)
            st.success("✅ Knowledge-Base loaded")

    else:
        st.info("Append logic unchanged")
