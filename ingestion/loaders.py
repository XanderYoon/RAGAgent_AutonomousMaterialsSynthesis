import fitz
import io
import base64
import mimetypes
import pandas as pd
import streamlit as st
from docx import Document as DocxDocument

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
