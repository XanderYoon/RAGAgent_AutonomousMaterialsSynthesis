import asyncio
import time
import socket
import json
import sys
import tempfile
from multiprocessing import Process
from pathlib import Path

import faiss
import numpy as np
import pickle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastmcp import Client  # noqa: E402
from RAG.mcp.app import create_app  # noqa: E402
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


def _run_server(port: int):
    app = create_app()
    app.run(transport="http", host="127.0.0.1", port=port)


async def _call_kb_load(port: int, index_path: Path, meta_path: Path):
    client = Client(f"http://127.0.0.1:{port}/mcp/")
    async with client:
        result = await client.call_tool(
            "kb_load",
            {
                "params": {
                    "index_path": str(index_path),
                    "meta_path": str(meta_path),
                    "graphrag_dir": None,
                }
            },
        )
    return result


def _extract_json_text(result):
    if isinstance(result, list) and result:
        content = result[0]
        text = getattr(content, "text", None)
        if text:
            return json.loads(text)
    return result


def main():
    embedding_model = "text-embedding-3-small"
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        index_path, meta_path = _write_dummy_kb(root, embedding_model)

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        proc = Process(target=_run_server, args=(port,), daemon=True)
        proc.start()

        try:
            for _ in range(10):
                try:
                    result = asyncio.run(
                        _call_kb_load(port, index_path, meta_path)
                    )
                    parsed = _extract_json_text(result)
                    print("MCP kb_load result:", parsed)
                    break
                except Exception:
                    time.sleep(0.2)
            else:
                raise RuntimeError("Server did not respond in time.")
        finally:
            proc.terminate()
            proc.join(timeout=2)
            if proc.is_alive():
                proc.kill()


if __name__ == "__main__":
    main()
