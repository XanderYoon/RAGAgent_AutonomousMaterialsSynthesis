from pathlib import Path

import pytest
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI

from RAG.services.kb_service import append_kb, build_kb, load_kb
from state.config import ENC


def _write_pdf(path: Path, text: str):
    import fitz

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    doc.save(str(path))
    doc.close()


def test_load_kb_dummy(dummy_kb):
    index_path, meta_path, _ = dummy_kb
    result = load_kb(
        index_path=str(index_path),
        meta_path=str(meta_path),
        graphrag_dir=None,
    )
    assert result["status"] == "ok"
    assert result["chunk_count"] == 1


@pytest.mark.integration
@pytest.mark.requires_openai
def test_kb_build_integration(tmp_path, openai_key):
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    _write_pdf(pdf_dir / "doc1.pdf", "Smoke test document for KB build.")

    index_path = tmp_path / "index.faiss"
    meta_path = tmp_path / "meta.pkl"
    graphrag_dir = tmp_path / "graphrag"

    client = OpenAI(api_key=openai_key)
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=openai_key,
    )

    result = build_kb(
        client=client,
        embeddings=embeddings,
        enc=ENC,
        pdf_dir=str(pdf_dir),
        index_path=str(index_path),
        meta_path=str(meta_path),
        graphrag_dir=str(graphrag_dir),
        embedding_model="text-embedding-3-small",
        run_graphrag=False,
        api_key=openai_key,
    )

    assert index_path.is_file()
    assert meta_path.is_file()
    assert result["total_chunks"] > 0


@pytest.mark.integration
@pytest.mark.requires_openai
def test_kb_append_integration(tmp_path, openai_key):
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    _write_pdf(pdf_dir / "doc1.pdf", "Initial KB document.")

    index_path = tmp_path / "index.faiss"
    meta_path = tmp_path / "meta.pkl"
    graphrag_dir = tmp_path / "graphrag"

    client = OpenAI(api_key=openai_key)
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=openai_key,
    )

    build_kb(
        client=client,
        embeddings=embeddings,
        enc=ENC,
        pdf_dir=str(pdf_dir),
        index_path=str(index_path),
        meta_path=str(meta_path),
        graphrag_dir=str(graphrag_dir),
        embedding_model="text-embedding-3-small",
        run_graphrag=False,
        api_key=openai_key,
    )

    append_dir = tmp_path / "append"
    append_dir.mkdir()
    _write_pdf(append_dir / "doc2.pdf", "Appended KB document.")

    result = append_kb(
        client=client,
        enc=ENC,
        index_path=str(index_path),
        meta_path=str(meta_path),
        append_folder=str(append_dir),
        graphrag_dir=str(graphrag_dir),
        run_graphrag=False,
        api_key=openai_key,
    )

    assert result["new_chunks"] > 0
