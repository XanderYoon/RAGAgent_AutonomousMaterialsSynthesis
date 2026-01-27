import streamlit as st
from RAG.retrieval.kb_builder import KnowledgeBaseBuilder

# -------------------------
# Knowledge-Base Setup UI
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
