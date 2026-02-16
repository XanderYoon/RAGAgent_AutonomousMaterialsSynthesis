import os
import pickle
from pathlib import Path

import faiss
import streamlit as st
from langchain_community.retrievers import BM25Retriever
from pydantic import ValidationError

from RAG.ingestion.faiss_store import load_faiss_from_disk
from RAG.kb_builder import (
    KnowledgeBaseError,
    append_kb,
    build_kb,
    load_kb,
)
from state.config import EMBEDDING_DIMENSIONS, TOP_K_TEXT_BM25
from state.schemas import KBAppendRequest, KBBuildRequest, KBLoadRequest


# -------------------------
# Knowledge-Base Setup UI
# -------------------------

def _dequote_path(path):
    """Strip accidental shell quotes."""
    if path is None:
        return path
    return path.strip().strip('"').strip("'")


def _register_kb_in_session(index_path: str, meta_path: str, graphrag_dir: str, embeddings):
    """Load KB artifacts from disk and register them in Streamlit session state."""
    index = faiss.read_index(index_path)
    with open(meta_path, "rb") as f:
        metadata = pickle.load(f)

    db, docs = load_faiss_from_disk(index, metadata, embeddings)
    model = metadata[0]["embedding_model"]

    st.session_state.update(
        index=index,
        metadata=metadata,
        db=db,
        all_documents=docs,
        bm25=BM25Retriever.from_documents(docs, k=TOP_K_TEXT_BM25),
        embedding_model=model,
        dimension=EMBEDDING_DIMENSIONS[model],
        graphrag=graphrag_dir,
        index_path=index_path,
        meta_path=meta_path,
    )


class StreamlitKBCallbacks:
    """Callback adapter used by core KB functions for UI progress updates."""

    def __init__(self):
        self._bars = {}

    def info(self, msg):
        st.write(msg)

    def warning(self, msg):
        st.warning(msg)

    def success(self, msg):
        st.success(msg)

    def progress(self, phase, current, total):
        if total <= 0:
            return
        if phase not in self._bars:
            self._bars[phase] = st.progress(0)
        pct = int(current / max(total, 1) * 100)
        self._bars[phase].progress(pct)


def kb_setup(client, embeddings):
    """Render knowledge-base setup controls and execute selected operation."""
    st.header("📂 Knowledge-Base Setup")
    has_valid_key = st.session_state.get("api_verified", False)
    if not has_valid_key:
        st.info("ℹ️ Please set a valid API key.")

    mode = st.radio(
        "Choose setup method:",
        ("📚 Build", "📤 Load", "➕ Append"),
    )

    if mode == "📚 Build":
        model = st.selectbox(
            "Embedding model",
            ["text-embedding-3-small", "text-embedding-3-large"],
            help="This model is used for generating embeddings -> impacts context matching",
        )
        pdf_dir = _dequote_path(
            st.text_input(
                "📁 Input Folder path for PDFs",
                value="inputs",
                help="Path to get the PDFs as input to build the Knowledge-Base",
                disabled=not has_valid_key,
            )
        )
        index_path = _dequote_path(
            st.text_input(
                "🧠 Output FAISS index file path (.index)",
                value="outputs/test_index.index",
                help="Path to save the FAISS index",
                disabled=not has_valid_key,
            )
        )
        meta_path = _dequote_path(
            st.text_input(
                "📝 Output metadata file path (.pkl)",
                value="outputs/test_metadata.pkl",
                help="Path to save metadata for chunks",
                disabled=not has_valid_key,
            )
        )
        graphrag_dir = _dequote_path(
            st.text_input(
                "📁 Parent directory for Knowledge-Graph",
                value="outputs/graphrag",
                help="GraphRAG workspace directory for input/output artifacts",
                disabled=not has_valid_key,
            )
        )

        if st.button("Build", disabled=not has_valid_key):
            try:
                _ = KBBuildRequest(
                    pdf_dir=pdf_dir,
                    index_path=index_path,
                    meta_path=meta_path,
                    graphrag_dir=graphrag_dir,
                    embedding_model=model,
                )
            except ValidationError as exc:
                st.error(f"Invalid build parameters: {exc}")
                st.stop()

            if not os.path.isdir(pdf_dir):
                st.error("The provided folder path does not exist!")
                st.stop()
            if not list(Path(pdf_dir).glob("*.pdf")):
                st.error("No PDF files found in the selected folder!")
                st.stop()

            callbacks = StreamlitKBCallbacks()
            try:
                _ = build_kb(
                    client=client,
                    embeddings=embeddings,
                    enc=st.session_state.enc,
                    pdf_dir=pdf_dir,
                    index_path=index_path,
                    meta_path=meta_path,
                    graphrag_dir=graphrag_dir,
                    embedding_model=model,
                    run_graphrag=True,
                    api_key=st.session_state.api_key,
                    callbacks=callbacks,
                )
                _register_kb_in_session(index_path, meta_path, graphrag_dir, embeddings)
            except KnowledgeBaseError as exc:
                st.error(str(exc))
                return

            st.success("✅ Knowledge-Base built")

    elif mode == "📤 Load":
        index_path = _dequote_path(
            st.text_input(
                "🧠 Index file path (.index)",
                value="outputs/test_index.index",
                disabled=not has_valid_key,
            )
        )
        meta_path = _dequote_path(
            st.text_input(
                "📝 Metadata file path (.pkl)",
                value="outputs/test_metadata.pkl",
                disabled=not has_valid_key,
            )
        )
        graphrag_dir = _dequote_path(
            st.text_input(
                "🕸️ Directory for Knowledge-Graph",
                value="outputs/graphrag",
                help="Folder where GraphRAG artifacts are stored",
                disabled=not has_valid_key,
            )
        )
        if st.button("Load", disabled=not has_valid_key):
            try:
                _ = KBLoadRequest(
                    index_path=index_path,
                    meta_path=meta_path,
                    graphrag_dir=graphrag_dir,
                )
            except ValidationError as exc:
                st.error(f"Invalid load parameters: {exc}")
                st.stop()

            if not os.path.isfile(index_path):
                st.error("Index file not found!")
                st.stop()
            if not os.path.isfile(meta_path):
                st.error("Metadata file not found!")
                st.stop()

            try:
                _ = load_kb(
                    index_path=index_path,
                    meta_path=meta_path,
                    graphrag_dir=graphrag_dir,
                )
                _register_kb_in_session(index_path, meta_path, graphrag_dir, embeddings)
            except KnowledgeBaseError as exc:
                st.error(str(exc))
                return

            st.success("✅ Knowledge-Base loaded")

    elif mode == "➕ Append":
        exist_index_path = _dequote_path(
            st.text_input(
                "🧠 Existing index file path (.index)",
                value="outputs/index.index",
                disabled=not has_valid_key,
            )
        )
        exist_meta_path = _dequote_path(
            st.text_input(
                "📝 Existing metadata file path (.pkl)",
                value="outputs/meta.pkl",
                disabled=not has_valid_key,
            )
        )
        graphrag_dir = _dequote_path(
            st.text_input(
                "🕸️ GraphRAG dir (existing workspace)",
                value="outputs/graphrag",
                disabled=not has_valid_key,
                help="Must contain input/ and output/ from a prior build.",
            )
        )
        append_folder = _dequote_path(
            st.text_input(
                "📁 Input Folder path for NEW PDFs to append:",
                value="inputs/new",
                help="Path to the NEW PDFs to be appended",
                disabled=not has_valid_key,
            )
        )

        if st.button("➕ Append to Knowledge-Base", disabled=not has_valid_key):
            try:
                _ = KBAppendRequest(
                    index_path=exist_index_path,
                    meta_path=exist_meta_path,
                    append_folder=append_folder,
                    graphrag_dir=graphrag_dir,
                )
            except ValidationError as exc:
                st.error(f"Invalid append parameters: {exc}")
                st.stop()

            if not os.path.isfile(exist_index_path):
                st.error("Existing FAISS index file not found!")
                st.stop()
            if not os.path.isfile(exist_meta_path):
                st.error("Existing metadata file not found!")
                st.stop()
            if not os.path.isdir(append_folder):
                st.error("Append folder does not exist!")
                st.stop()

            callbacks = StreamlitKBCallbacks()
            try:
                _ = append_kb(
                    client=client,
                    enc=st.session_state.enc,
                    index_path=exist_index_path,
                    meta_path=exist_meta_path,
                    append_folder=append_folder,
                    graphrag_dir=graphrag_dir,
                    run_graphrag=True,
                    api_key=st.session_state.api_key,
                    callbacks=callbacks,
                )
                _register_kb_in_session(
                    exist_index_path,
                    exist_meta_path,
                    graphrag_dir,
                    embeddings,
                )
            except KnowledgeBaseError as exc:
                st.error(str(exc))
                return

            st.success("✅ Knowledge-Base appended")
