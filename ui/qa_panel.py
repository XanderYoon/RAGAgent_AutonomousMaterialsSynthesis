import time
import os
import subprocess
from pathlib import Path

import numpy as np
import streamlit as st
from docx import Document as DocxDocument
from langchain.retrievers import EnsembleRetriever, MultiQueryRetriever
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_community.document_transformers import LongContextReorder
from langchain_openai import ChatOpenAI

from config import (
    TOP_K_TEXT_FAISS,
    TOP_K_UPLOAD_FAISS,
    STREAM_DELAY,
)

from ingestion.loaders import _to_data_url
from ingestion.loaders import _parse_table_file
from ingestion.loaders import extract_text_from_pdf
from ingestion.cleaners import remove_junk_sections, remove_junk_lines
from ingestion.chunkers import chunk_text2
from ingestion.faiss_store import load_faiss_from_disk
from ingestion.graphrag import run_graphrag_cli

# ==============================
# Model selection
# ==============================
def model_selector():
    st.session_state.gpt_model = st.selectbox(
        "🤖 Select model:",
        options=[
            "gpt-4.1-2025-04-14",
            "gpt-4o-2024-08-06",
            "gpt-5",
            "gpt-5-thinking",
            "gpt-5-pro",
            "o4-mini-2025-04-16",
            "o4-mini-deep-research-2025-06-26",
        ],
        index=0,
        help="This model generates the final answer!",
    )

# ==============================
# Advanced controls
# ==============================
def advanced_controls():
    with st.expander("🎛️ Advanced Controls: Diversity & Creativity"):
        st.session_state.diversity = st.slider(
            "🧭 Diversity Radar (0 = Homogeneous, 1 = Diverse)",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.01,
        )

        if st.session_state.gpt_model in [
            "gpt-4o-2024-08-06",
            "gpt-4.1-2025-04-14",
        ]:
            st.session_state.temperature = st.slider(
                "🔥 Creativity Dial (0 = Boring, 1 = Creative)",
                min_value=0.0,
                max_value=1.0,
                value=0.3,
                step=0.01,
            )
        else:
            st.session_state.temperature = 0.3


def qa_panel(client):
    st.header("❓ Ask a Question")

    model_selector()
    advanced_controls()

    if "index" not in st.session_state:
        st.info("⚠️ Please build or load a Knowledge-Base before asking a question!")
        return

    for key in ["last_query", "last_answer", "context_meta"]:
        if key not in st.session_state:
            st.session_state[key] = ""

    query = st.text_area(
        "Ask your question here:",
        height=280,
        placeholder="Type your question...",
    )

    # ------------------------------
    # Upload UI
    # ------------------------------
    st.markdown("#### 📎 Add any relevant file(s) for this question (optional)")
    uploaded_files = st.file_uploader(
        "Upload PDFs/TXT/CSV/XLSX or images:",
        type=["pdf", "txt", "csv", "xlsx", "png", "jpg", "jpeg", "gif", "bmp", "tif", "tiff"],
        accept_multiple_files=True,
    )

    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        answer_clicked = st.button("💬 Answer", use_container_width=True)
    with col3:
        save_clicked = st.button("💾 Save last Q&A", use_container_width=True)

    # ------------------------------
    # ANSWER
    # ------------------------------
    if answer_clicked and query:
        with st.spinner("🔍 Retrieving context..."):
            query_embedding = client.embeddings.create(
                input=query,
                model=st.session_state.embedding_model,
            ).data[0].embedding

            vs_retriever = st.session_state.db.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": TOP_K_TEXT_FAISS,
                    "fetch_k": TOP_K_TEXT_FAISS * 2,
                    "lambda_mult": 1.0 - st.session_state.diversity,
                },
            )

            hybrid = EnsembleRetriever(
                retrievers=[vs_retriever, st.session_state.bm25],
                weights=[0.5, 0.5],
            )

            llm_expander = ChatOpenAI(model="gpt-4o-mini")
            multi_query = MultiQueryRetriever.from_llm(
                retriever=hybrid,
                llm=llm_expander,
                include_original=True,
            )

            compressor = LLMChainExtractor.from_llm(
                ChatOpenAI(model="gpt-4o-mini", temperature=0)
            )

            retriever = ContextualCompressionRetriever(
                base_retriever=multi_query,
                base_compressor=compressor,
            )

            results = retriever.invoke(query)
            results = LongContextReorder().transform_documents(results)

            context_meta_chunks = [
                f"[{d.metadata['source']} | chunk {d.metadata['chunk_id']}]: {d.page_content}"
                for d in results
            ]

        context_meta = "\n\n".join(context_meta_chunks)

        # ------------------------------
        # GENERATION
        # ------------------------------
        system_instructions = (
            "You are an expert scientific research assistant. "
            "Use the provided context to answer accurately."
        )

        prompt = f"""
User query: {query}

--- BEGIN CONTEXT ---
{context_meta}
--- END CONTEXT ---

Answer:
""".strip()

        placeholder = st.empty()
        answer_text = ""

        with client.chat.completions.stream(
            model=st.session_state.gpt_model,
            messages=[
                {"role": "system", "content": system_instructions},
                {"role": "user", "content": prompt},
            ],
            temperature=st.session_state.temperature,
        ) as stream:
            for event in stream:
                if event.type == "content.delta":
                    answer_text += event.delta
                    placeholder.markdown(answer_text + "▌")
                    time.sleep(STREAM_DELAY)
                elif event.type == "content.done":
                    placeholder.markdown(answer_text)

        st.session_state.last_query = query
        st.session_state.last_answer = answer_text
        st.session_state.context_meta = context_meta

    # ------------------------------
    # SAVE Q&A
    # ------------------------------
    if save_clicked and st.session_state.last_answer:
        save_path = Path("outputs/qa_pairs.docx")
        doc = DocxDocument(save_path) if save_path.exists() else DocxDocument()
        doc.add_heading("Question:", level=2)
        doc.add_paragraph(st.session_state.last_query)
        doc.add_heading("Answer:", level=2)
        doc.add_paragraph(st.session_state.last_answer)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        doc.save(save_path)
        st.success(f"✅ Saved to {save_path}")

    # ------------------------------
    # CONTEXT VIEW
    # ------------------------------
    if st.session_state.context_meta:
        st.divider()
        with st.expander("📚 Retrieved Context"):
            st.markdown(
                st.session_state.context_meta.replace("\n", "<br>"),
                unsafe_allow_html=True,
            )