import time
from pathlib import Path

import streamlit as st
from docx import Document as DocxDocument
from langchain.retrievers import EnsembleRetriever, MultiQueryRetriever
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_community.document_transformers import LongContextReorder
from langchain_openai import ChatOpenAI

from config import TOP_K_TEXT_FAISS, STREAM_DELAY


# -------------------------
# Controls
# -------------------------

DEFAULT_MODEL_OPTIONS = [
    "gpt-4.1-2025-04-14",
    "gpt-4o-2024-08-06",
    "gpt-5",
    "gpt-5-thinking",
    "gpt-5-pro",
    "o4-mini-2025-04-16",
    "o4-mini-deep-research-2025-06-26",
]


def _is_allowed_model(model_id):
    """Allow o* and non-codex GPT models."""
    mid = (model_id or "").lower()
    return mid.startswith("o") or (mid.startswith("gpt-") and "codex" not in mid)


def _load_model_options(client):
    """Fetch and cache available model IDs."""
    cached = st.session_state.get("available_models")
    if cached:
        return cached

    if client is None:
        return DEFAULT_MODEL_OPTIONS

    try:
        data = client.models.list()
        models = [m.id for m in getattr(data, "data", [])]
        options = sorted({m for m in models if _is_allowed_model(m)})
        if options:
            st.session_state.available_models = options
            return options
    except Exception:
        pass

    return DEFAULT_MODEL_OPTIONS


def model_selector(options):
    """Select LLM for answer generation."""
    current = st.session_state.get("gpt_model")
    index = options.index(current) if current in options else 0
    st.session_state.gpt_model = st.selectbox(
        "🤖 Select model:",
        options,
        index=index,
    )


def advanced_controls():
    """Diversity + creativity controls."""
    with st.expander("🎛️ Advanced Controls"):
        st.session_state.diversity = st.slider(
            "Diversity", 0.0, 1.0, 0.7, 0.01
        )

        if st.session_state.gpt_model in {
            "gpt-4o-2024-08-06",
            "gpt-4.1-2025-04-14",
        }:
            st.session_state.temperature = st.slider(
                "Creativity", 0.0, 1.0, 0.3, 0.01
            )
        else:
            st.session_state.temperature = 0.3


# -------------------------
# Retrieval + Generation
# -------------------------

class QAEngine:
    """Retrieve context and generate answers."""

    def __init__(self, client):
        self.client = client

    def retrieve(self, query: str):
        vs = st.session_state.db.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": TOP_K_TEXT_FAISS,
                "fetch_k": TOP_K_TEXT_FAISS * 2,
                "lambda_mult": 1.0 - st.session_state.diversity,
            },
        )

        hybrid = EnsembleRetriever(
            retrievers=[vs, st.session_state.bm25],
            weights=[0.5, 0.5],
        )

        mq = MultiQueryRetriever.from_llm(
            retriever=hybrid,
            llm=ChatOpenAI(model="gpt-4o-mini"),
            include_original=True,
        )

        compressor = LLMChainExtractor.from_llm(
            ChatOpenAI(model="gpt-4o-mini", temperature=0)
        )

        retriever = ContextualCompressionRetriever(
            base_retriever=mq,
            base_compressor=compressor,
        )

        docs = retriever.invoke(query)
        docs = LongContextReorder().transform_documents(docs)

        return "\n\n".join(
            f"[{d.metadata['source']} | chunk {d.metadata['chunk_id']}]: {d.page_content}"
            for d in docs
        )

    def generate(self, query: str, context: str):
        system = (
            "You are an expert scientific research assistant. "
            "Use the provided context to answer accurately."
        )

        prompt = f"""
User query: {query}

--- BEGIN CONTEXT ---
{context}
--- END CONTEXT ---

Answer:
""".strip()

        placeholder = st.empty()
        answer = ""

        with self.client.chat.completions.stream(
            model=st.session_state.gpt_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            temperature=st.session_state.temperature,
        ) as stream:
            for event in stream:
                if event.type == "content.delta":
                    answer += event.delta
                    placeholder.markdown(answer + "▌")
                    time.sleep(STREAM_DELAY)
                elif event.type == "content.done":
                    placeholder.markdown(answer)

        return answer


# -------------------------
# UI
# -------------------------
def qa_panel(client):
    st.header("❓ Ask a Question")

    has_key = st.session_state.get("api_verified", False)
    has_kb = st.session_state.get("index") is not None
    disabled = not (has_key and has_kb)

    if not has_key:
        st.info("ℹ️ Please set a valid API key.")
    elif not has_kb:
        st.info("ℹ️ Please build or load a Knowledge-Base.")

    # ---- defaults
    st.session_state.setdefault("gpt_model", "gpt-4.1-2025-04-14")
    st.session_state.setdefault("last_query", "")
    st.session_state.setdefault("last_answer", "")
    st.session_state.setdefault("context_meta", "")

    model_selector(_load_model_options(client))
    advanced_controls()

    query = st.text_area(
        "Ask your question:",
        height=280,
        disabled=disabled,
    )

    st.markdown("#### 📎 Add any relevant file(s) for this question (optional)")
    uploaded_files = st.file_uploader(
        "Upload PDFs/TXT/CSV/XLSX or images:",
        type=[
            "pdf", "txt", "csv", "xlsx",
            "png", "jpg", "jpeg", "gif", "bmp", "tif", "tiff",
        ],
        accept_multiple_files=True,
        disabled=disabled,
    )
    
    col1, _, col3 = st.columns([1, 2, 1])
    answer_clicked = col1.button(
        "💬 Answer", use_container_width=True, disabled=disabled
    )
    save_clicked = col3.button(
        "💾 Save last Q&A", use_container_width=True, disabled=disabled
    )

    engine = QAEngine(client)

    if answer_clicked and query:
        with st.spinner("🔍 Retrieving context..."):
            context = engine.retrieve(query)

        answer = engine.generate(query, context)

        st.session_state.update(
            last_query=query,
            last_answer=answer,
            context_meta=context,
        )

    if save_clicked and st.session_state.last_answer:
        path = Path("outputs/qa_pairs.docx")
        doc = DocxDocument(path) if path.exists() else DocxDocument()
        doc.add_heading("Question:", 2)
        doc.add_paragraph(st.session_state.last_query)
        doc.add_heading("Answer:", 2)
        doc.add_paragraph(st.session_state.last_answer)
        path.parent.mkdir(parents=True, exist_ok=True)
        doc.save(path)
        st.success(f"✅ Saved to {path}")

    if st.session_state.context_meta:
        st.divider()
        with st.expander("📚 Retrieved Context"):
            st.markdown(
                st.session_state.context_meta.replace("\n", "<br>"),
                unsafe_allow_html=True,
            )
