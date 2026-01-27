import fitz
import io
import base64
import mimetypes
import pandas as pd
import streamlit as st
from langchain_openai import OpenAIEmbeddings
from ingestion.cleaners import remove_junk_sections, remove_junk_lines
from ingestion.chunkers import chunk_text2
import numpy as np
from config import (
    TOKENS_PER_CHUNK,
    WORDS_PER_CHUNK_OVERLAP
)
from langchain_community.vectorstores import FAISS

from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.documents import Document
import faiss
from config import ENC

# ------------------------------
# PDF
# ------------------------------
def extract_text_from_pdf(path):
    doc = fitz.open(path)
    return "\n".join(page.get_text() for page in doc)

# ------------------------------
# Tables (CSV / XLSX)
# ------------------------------
def _parse_table_file(file_bytes, filename, max_rows=50, max_chars=20000):
    try:
        if filename.lower().endswith(".csv"):
            df = pd.read_csv(io.BytesIO(file_bytes))
        else:
            df = pd.read_excel(io.BytesIO(file_bytes), engine="openpyxl")
    except Exception as e:
        return f"[PARSE-ERROR {filename}: {e}]"

    buf = []
    buf.append(f"TABLE: {filename}")
    buf.append("COLUMNS: " + ", ".join(map(str, df.columns.tolist())))
    head = df.head(max_rows)
    buf.append("HEAD:")
    buf.append(head.to_csv(index=False))

    text = "\n".join(buf)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n...[truncated]..."

    return text

# ------------------------------
# Images
# ------------------------------
def _to_data_url(file_bytes, mime_type="image/png"):
    b64 = base64.b64encode(file_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{b64}"



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
            text, max_tokens=TOKENS_PER_CHUNK, tokenizer=ENC, overlap=WORDS_PER_CHUNK_OVERLAP
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