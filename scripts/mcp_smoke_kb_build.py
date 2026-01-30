import os
import sys
import tempfile
from pathlib import Path

import fitz
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from RAG.services.kb_service import build_kb  # noqa: E402
from state.config import ENC  # noqa: E402


def _require_api_key():
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise SystemExit("OPENAI_API_KEY is required for this smoke test.")
    return key


def _write_pdf(path: Path, text: str):
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    doc.save(str(path))
    doc.close()


def main():
    api_key = _require_api_key()
    client = OpenAI(api_key=api_key)
    embedding_model = "text-embedding-3-small"
    embeddings = OpenAIEmbeddings(model=embedding_model, api_key=api_key)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        pdf_dir = root / "pdfs"
        pdf_dir.mkdir()
        _write_pdf(pdf_dir / "doc1.pdf", "Smoke test document for KB build.")

        index_path = root / "index.faiss"
        meta_path = root / "meta.pkl"
        graphrag_dir = root / "graphrag"

        result = build_kb(
            client=client,
            embeddings=embeddings,
            enc=ENC,
            pdf_dir=str(pdf_dir),
            index_path=str(index_path),
            meta_path=str(meta_path),
            graphrag_dir=str(graphrag_dir),
            embedding_model=embedding_model,
            run_graphrag=False,
            api_key=api_key,
        )

        print("KB build result:", result)
        print("Index exists:", index_path.is_file())
        print("Metadata exists:", meta_path.is_file())


if __name__ == "__main__":
    main()
