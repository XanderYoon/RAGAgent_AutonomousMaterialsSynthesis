import os
from pathlib import Path
import streamlit as st
from RAG.retrieval.kb_builder import KnowledgeBaseBuilder, KnowledgeBaseError
from pydantic import ValidationError

from state.schemas import KBBuildRequest, KBLoadRequest, KBAppendRequest

# -------------------------
# Knowledge-Base Setup UI
# -------------------------

def _dequote_path(path):
    """Strip accidental shell quotes."""
    if path is None:
        return path
    return path.strip().strip('"').strip("'")


class StreamlitKBCallbacks:
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

    def spinner(self, msg):
        return st.spinner(msg)


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
            help="This model is used for generating embeddings → impacts context matching"
        )
        pdf_dir = _dequote_path(
            st.text_input(
                "📁 Input Folder path for PDFs",
                value="inputs", 
                help="Path to get the PDFs as input to build the Knowledge-Base",
                disabled = not has_valid_key
            )
        )
        index_path = _dequote_path(
            st.text_input(
                "🧠 Output FAISS index file path (.index)",
                value='outputs/test_index.index',
                help="Path to save the FAISS index",
                disabled = not has_valid_key
            )
        )
        meta_path = _dequote_path(
            st.text_input(
                "📝 Output metadata file path (.pkl)",
                value='outputs/test_metadata.pkl',
                help="Path to save metadata for chunks",
                disabled = not has_valid_key
            )
        )
        graphrag_dir = _dequote_path(
            st.text_input(
                "📁 Parent directory for Knowledge-Graph",
                value='outputs',
                help="This is where the Knowledge-Graph pipeline will create 'knowledge_graph' subfolder and save related artifacts",
                disabled = not has_valid_key
            )
        )

        if st.button("Build",disabled = not has_valid_key):
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
            # Sanity checks
            if not os.path.isdir(pdf_dir):
                st.error("The provided folder path does not exist!")
                st.stop()
            pdf_files = list(Path(pdf_dir).glob("*.pdf"))
            if not pdf_files:
                st.error("No PDF files found in the selected folder!")
                st.stop()

            callbacks = StreamlitKBCallbacks()
            try:
                kb.build(pdf_dir, index_path, meta_path, graphrag_dir, model, callbacks)
            except KnowledgeBaseError as exc:
                st.error(str(exc))
                return

            st.success("✅ Knowledge-Base built")

    elif mode == "📤 Load":
        index_path = _dequote_path(
            st.text_input(
                "🧠 Index file path (.index)",
                value='outputs/test_index.index',
                disabled = not has_valid_key
            )
        )        
        meta_path = _dequote_path(
            st.text_input(
                "📝 Metadata file path (.pkl)",
                value='outputs/test_metadata.pkl',
                disabled = not has_valid_key
            )
        )
        graphrag_dir = _dequote_path(
            st.text_input(
                "🕸️ Directory for Knowledge-Graph",
                value='outputs/graphrag',
                help="Folder where the Knowledge-Graph pipeline saved related artifacts",
                disabled = not has_valid_key
            )
        )
        if st.button("Load", disabled = not has_valid_key):
            try:
                _ = KBLoadRequest(
                    index_path=index_path,
                    meta_path=meta_path,
                    graphrag_dir=graphrag_dir,
                )
            except ValidationError as exc:
                st.error(f"Invalid load parameters: {exc}")
                st.stop()
            # Sanity checks
            if not os.path.isdir(graphrag_dir):
                st.error("The provided folder path does not exist!")
                st.stop()
            if not os.path.isfile(index_path):
                st.error("Index file not found!")
                st.stop()
            if not os.path.isfile(meta_path):
                st.error("Metadata file not found!")
                st.stop()

            try:
                kb.load(index_path, meta_path, graphrag_dir)
            except KnowledgeBaseError as exc:
                st.error(str(exc))
                return

            required = [
                "entities.parquet",
                "relationships.parquet",
                "documents.parquet",
                "communities.parquet",
                "community_reports.parquet",
            ]

            missing = [
                f for f in required
                if not (Path(graphrag_dir) / "output" / f).exists()
            ]

            if missing:
                st.warning(
                    "⚠️ Knowledge-Graph directory is corrupted - missing critical files!"
                )
            else:
                st.session_state.graphrag = graphrag_dir
                st.success("✅ Knowledge-Graph loaded successfully!")


    elif mode == "➕ Append":
        exist_index_path = st.text_input(
            "🧠 Existing index file path (.index)",
            value="outputs/index.index",
            disabled=not has_valid_key,
        )
        exist_meta_path = st.text_input(
            "📝 Existing metadata file path (.pkl)",
            value="outputs/meta.pkl",
            disabled=not has_valid_key,
        )
        graphrag_dir = st.text_input(
            "🕸️ GraphRAG dir (existing workspace)",
            value="outputs/graphrag",
            disabled=not has_valid_key,
            help="Must contain input/ and output/ from a prior build.",
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
            # Sanity checks
            if not os.path.isfile(exist_index_path):
                st.error("Existing FAISS index file not found!")
                st.stop()
            if not os.path.isfile(exist_meta_path):
                st.error("Existing metadata file not found!")
                st.stop()
            if not os.path.isdir(append_folder):
                st.error("Append folder does not exist!")
                st.stop()
            if not os.path.isdir(graphrag_dir):
                st.error("Knowledge-Graph directory not found!")
                st.stop()

            callbacks = StreamlitKBCallbacks()
            try:
                kb.append(
                    exist_index_path,
                    exist_meta_path,
                    append_folder,
                    graphrag_dir,
                    callbacks,
                )
            except KnowledgeBaseError as exc:
                st.error(str(exc))
                return
