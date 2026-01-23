from config import (
    EMBEDDING_DIMENSIONS,
    TOKENS_PER_CHUNK,
    WORDS_PER_CHUNK_OVERLAP,
    TOP_K_TEXT_FAISS,
    TOP_K_TEXT_BM25,
    TOP_K_UPLOAD_FAISS,
    TOP_K_GRAPH,
    STREAM_DELAY,
)

from ingestion.cleaners import (
    remove_junk_sections,
    remove_junk_lines,
)

from ingestion.chunkers import (
    chunk_text,
    chunk_text2,
)

from ingestion.loaders import (
    extract_text_from_pdf,
    _parse_table_file,
    _to_data_url,
)

from ingestion.faiss_store import (
    build_faiss_from_embeddings,
    load_faiss_from_disk,
)

from ingestion.embeddings import estimate_embedding_cost

from ingestion.graphrag import (
    run_graphrag_cli,
    StaticGraphRetriever,
)

from ui.API_wall import API_entry
from ui.kb_setup import kb_setup

from ui.qa_panel import (
    model_selector,
    advanced_controls,
    qa_panel
)

from pathlib import Path
import re
import fitz
import tiktoken
from openai import OpenAI
import numpy as np
import faiss
import streamlit as st
import pickle
import os
import openai
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pycountry
import time
from langchain_community.vectorstores import FAISS 
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
import pandas as pd
import io
import base64
import mimetypes
from docx import Document as DocxDocument
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever, MultiQueryRetriever
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.document_transformers import LongContextReorder
from langchain_experimental.graph_transformers import LLMGraphTransformer
import pickle
import networkx as nx
from langchain_core.retrievers import BaseRetriever
from langchain.globals import set_llm_cache
from langchain_community.cache import SQLiteCache
import json
import subprocess
import yaml


################################
# Helpers 
################################

def dequote_path(p: str) -> str:
    if not p:
        return p
    p = p.strip()
    if len(p) >= 2 and p[0] == p[-1] and p[0] in "\"'":
        p = p[1:-1]
    return p


def build_upload_bundle(uploaded_files, client, embedding_model, dimension):
    """
    Build in-memory FAISS for uploaded *text-like* files and collect images.
    Returns: (upload_db, text_meta, images)
      - upload_db: FAISS store for uploaded text chunks (or None if none)
      - text_meta: list of dicts for chunks
      - images: list of {"name": str, "data_url": str}
    """
    text_chunks = []
    text_meta = []
    images = []

    for uf in uploaded_files:
        name = uf.name
        mime = uf.type or mimetypes.guess_type(name)[0] or ""
        data = uf.getvalue()  # bytes

        # Images
        if name.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tif", ".tiff")):
            data_url = _to_data_url(data, mime or "image/png")
            images.append({"name": name, "data_url": data_url})
            continue

        # Text-like docs
        text = ""
        if name.lower().endswith(".pdf"):
            try:
                with fitz.open(stream=data, filetype="pdf") as doc:
                    text = "\n".join(page.get_text() for page in doc)
            except Exception as e:
                st.warning(f"⚠️ Could not read PDF {name}: {e}")
                continue
        elif name.lower().endswith(".txt"):
            text = data.decode("utf-8", errors="ignore")
        elif name.lower().endswith(".csv") or name.lower().endswith(".xlsx"):
            text = _parse_table_file(data, name)
        else:
            st.warning(f"⚠️ Unsupported file type: {name} (supported: pdf/txt/csv/xlsx + images)")
            continue

        # Clean + chunk
        text = remove_junk_sections(text)
        text = remove_junk_lines(text)
        chunks = chunk_text2(
            text, max_tokens=TOKENS_PER_CHUNK, tokenizer=enc, overlap=WORDS_PER_CHUNK_OVERLAP
        )
        for i, ch in enumerate(chunks):
            text_chunks.append(ch)
            text_meta.append({
                "source": f"uploaded/{name}",
                "chunk_id": i,
                "text": ch,
                "embedding_model": embedding_model
            })

    # Build FAISS for uploaded text
    upload_db = None
    if text_chunks:
        BATCH = 64
        embs = []
        for i in range(0, len(text_chunks), BATCH):
            batch = text_chunks[i:i + BATCH]
            resp = client.embeddings.create(input=batch, model=embedding_model)
            embs.extend([d.embedding for d in resp.data])

        emb_mat = np.array(embs, dtype="float32")
        index = faiss.IndexFlatL2(dimension)
        index.add(emb_mat)

        ids = [str(i) for i in range(len(text_meta))]
        docs_dict = {
            ids[i]: Document(
                page_content=text_meta[i]["text"],
                metadata={
                    "source": text_meta[i]["source"], 
                    "chunk_id": text_meta[i]["chunk_id"],
                    "original_content": text_meta[i]["text"], 
                }
            ) for i in range(len(text_meta))
        }
        docstore = InMemoryDocstore(docs_dict)
        index_to_docstore_id = {i: ids[i] for i in range(len(ids))}

        upload_db = FAISS(
            embedding_function=OpenAIEmbeddings(model=embedding_model,api_key=st.session_state.api_key),
            index=index,
            docstore=docstore,
            index_to_docstore_id=index_to_docstore_id,
        )

    return upload_db, text_meta, images
    
################################
# Globals / Config
################################

enc = tiktoken.get_encoding("cl100k_base")

################################
# App UI
################################
st.markdown("## 📄 RAG Agent — Autonomous Synthesis")
st.divider()

API_entry()
client = st.session_state.client
st.divider()

if (st.session_state.api_key):
    embeddings_obj = OpenAIEmbeddings(
        model=st.session_state.get("embedding_model", "text-embedding-3-small"),
        api_key=st.session_state.api_key,
    )

    kb_setup(
        client=st.session_state.client,
        embeddings_obj=embeddings_obj,
    )

st.divider()

qa_panel(client=st.session_state.client)
