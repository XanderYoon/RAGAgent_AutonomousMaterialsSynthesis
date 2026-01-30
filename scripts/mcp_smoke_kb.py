import sys
import tempfile
from pathlib import Path

import faiss
import numpy as np
import pickle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from RAG.services.kb_service import load_kb  # noqa: E402
from state.config import EMBEDDING_DIMENSIONS  # noqa: E402


def _write_dummy_kb(root: Path, embedding_model: str):
    dim = EMBEDDING_DIMENSIONS[embedding_model]
    index = faiss.IndexFlatL2(dim)
    index.add(np.zeros((1, dim), dtype="float32"))

    index_path = root / "index.faiss"
    meta_path = root / "meta.pkl"

    faiss.write_index(index, str(index_path))
    metadata = [
        {
            "source": "dummy.pdf",
            "chunk_id": 0,
            "text": "dummy",
            "embedding_model": embedding_model,
        }
    ]
    meta_path.write_bytes(pickle.dumps(metadata))

    return index_path, meta_path


def main():
    embedding_model = "text-embedding-3-small"
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        index_path, meta_path = _write_dummy_kb(root, embedding_model)

        result = load_kb(
            index_path=str(index_path),
            meta_path=str(meta_path),
            graphrag_dir=None,
        )
        print("KB load result:", result)


if __name__ == "__main__":
    main()
