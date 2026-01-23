# ui/kb_setup.py
import os
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


def kb_setup(client, embeddings_obj):
    st.header("📂 Knowledge-Base Setup")

    option = st.radio(
        "Choose how you want to set up the Knowledge-Base:",
        (
            "📚 Build a new Knowledge-Base",
            "📤 Load existing Knowledge-Base",
            "➕ Append existing Knowledge-Base",
        ),
        index=0,
    )

    # ==========================================================
    # BUILD KB
    # ==========================================================
    if option == "📚 Build a new Knowledge-Base":
        st.session_state.embedding_model = st.selectbox(
            "🔍 Select your embedding model:",
            options=["text-embedding-3-small", "text-embedding-3-large"],
            index=0,
        )
        st.session_state.dimension = EMBEDDING_DIMENSIONS[
            st.session_state.embedding_model
        ]

        pdf_folder = st.text_input("📁 Input Folder path for PDFs:", value="inputs")
        index_path = st.text_input(
            "🧠 Output FAISS index file path (.index)",
            value="outputs/test_index.index",
        )
        meta_path = st.text_input(
            "📝 Output metadata file path (.pkl)",
            value="outputs/test_metadata.pkl",
        )
        graphrag_dir = st.text_input(
            "📁 Directory for Knowledge-Graph",
            value="outputs/graphrag",
        )

        if st.button("📚 Build Knowledge-Base"):
            pdf_files = list(Path(pdf_folder).glob("*.pdf"))
            if not pdf_files:
                st.error("No PDF files found!")
                st.stop()

            all_chunks = {}
            all_texts = {}
            total_chunks = total_tokens = total_cost = 0

            for pdf_path in pdf_files:
                text = extract_text_from_pdf(pdf_path)
                text = remove_junk_sections(text)
                text = remove_junk_lines(text)
                all_texts[pdf_path.name] = text

                chunks = chunk_text2(
                    text,
                    max_tokens=TOKENS_PER_CHUNK,
                    tokenizer=st.session_state.enc,
                    overlap=WORDS_PER_CHUNK_OVERLAP,
                )

                token_count = sum(len(st.session_state.enc.encode(c)) for c in chunks)
                total_chunks += len(chunks)
                total_tokens += token_count
                total_cost += estimate_embedding_cost(
                    token_count, st.session_state.embedding_model
                )
                all_chunks[pdf_path.name] = chunks

            all_embeddings = []
            all_metadata = []

            for fname, chunks in all_chunks.items():
                for i, chunk in enumerate(chunks):
                    emb = client.embeddings.create(
                        input=chunk,
                        model=st.session_state.embedding_model,
                    ).data[0].embedding

                    all_embeddings.append(emb)
                    all_metadata.append(
                        {
                            "source": fname,
                            "chunk_id": i,
                            "text": chunk,
                            "embedding_model": st.session_state.embedding_model,
                        }
                    )

            index, db, all_documents = build_faiss_from_embeddings(
                all_embeddings,
                all_metadata,
                embeddings_obj,
                st.session_state.dimension,
            )

            Path(index_path).parent.mkdir(parents=True, exist_ok=True)
            Path(meta_path).parent.mkdir(parents=True, exist_ok=True)

            faiss.write_index(index, index_path)
            with open(meta_path, "wb") as f:
                pickle.dump(all_metadata, f)

            st.session_state.index = index
            st.session_state.metadata = all_metadata
            st.session_state.db = db
            st.session_state.all_documents = all_documents
            st.session_state.bm25 = BM25Retriever.from_documents(
                all_documents, k=TOP_K_TEXT_BM25
            )

            # GraphRAG
            root = Path(graphrag_dir)
            (root / "input").mkdir(parents=True, exist_ok=True)
            (root / "output").mkdir(parents=True, exist_ok=True)

            for fname, text in all_texts.items():
                with open(root / "input" / f"{Path(fname).stem}.txt", "w") as f:
                    f.write(text)

            run_graphrag_cli(root, root / "input", root / "output", st.session_state.api_key)
            st.session_state.graphrag = graphrag_dir

            st.success("✅ Knowledge-Base built successfully!")

    # ==========================================================
    # LOAD KB
    # ==========================================================
    elif option == "📤 Load existing Knowledge-Base":
        index_path = st.text_input("🧠 Index file path (.index)")
        meta_path = st.text_input("📝 Metadata file path (.pkl)")
        graphrag_dir = st.text_input("📁 Knowledge-Graph directory")

        if st.button("📤 Load Knowledge-Base"):
            index = faiss.read_index(index_path)
            with open(meta_path, "rb") as f:
                metadata = pickle.load(f)

            st.session_state.embedding_model = metadata[0]["embedding_model"]
            st.session_state.dimension = EMBEDDING_DIMENSIONS[
                st.session_state.embedding_model
            ]

            db, all_documents = load_faiss_from_disk(
                index, metadata, embeddings_obj
            )

            st.session_state.index = index
            st.session_state.metadata = metadata
            st.session_state.db = db
            st.session_state.all_documents = all_documents
            st.session_state.bm25 = BM25Retriever.from_documents(
                all_documents, k=TOP_K_TEXT_BM25
            )
            st.session_state.graphrag = graphrag_dir

            st.success("✅ Knowledge-Base loaded!")

    # ==========================================================
    # APPEND KB
    # ==========================================================
    else:
        st.info("Append logic unchanged — moved verbatim in next step")
