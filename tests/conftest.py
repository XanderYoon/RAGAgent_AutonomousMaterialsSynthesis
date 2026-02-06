import os
import pickle
import socket
import sys
import time
from multiprocessing import Process
from pathlib import Path

import faiss
import fitz
import numpy as np
import pytest
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from RAG.mcp.app import create_app  # noqa: E402
from RAG.services.kb_service import build_kb  # noqa: E402
from state.config import EMBEDDING_DIMENSIONS, ENC  # noqa: E402


def _report_has_warnings(report: pytest.TestReport) -> bool:
    for name, _content in report.sections:
        if "warning" in name.lower():
            return True
    return False


def pytest_configure(config):
    config._passed_with_warnings = []


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if report.when == "call" and report.passed:
        if _report_has_warnings(report):
            item.config._passed_with_warnings.append(report.nodeid)


def pytest_terminal_summary(terminalreporter):
    stats = terminalreporter.stats
    total_collected = getattr(terminalreporter, "_numcollected", 0) or 0
    passed_count = len(stats.get("passed", []))
    failed_count = len(stats.get("failed", [])) + len(stats.get("error", []))
    if total_collected:
        pass_rate = (passed_count / total_collected) * 100
        line = f"Pass rate: {pass_rate:.0f}%"
        if failed_count == 0 and passed_count == total_collected:
            terminalreporter.write_line(
                f"{line} [PASS]",
                green=True,
            )
        else:
            terminalreporter.write_line(line)

    passed_with_warnings = getattr(
        terminalreporter.config, "_passed_with_warnings", []
    )
    if not passed_with_warnings:
        return

    terminalreporter.section("Passed With Warnings")


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


def _write_pdf(path: Path, text: str):
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    doc.save(str(path))
    doc.close()


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _wait_for_port(port: int, timeout: float = 5.0):
    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                return
        except OSError:
            time.sleep(0.1)
    raise RuntimeError(f"Server did not start on port {port}.")


def _run_server(port: int):
    app = create_app()
    app.run(transport="http", host="127.0.0.1", port=port)


@pytest.fixture
def dummy_kb(tmp_path):
    embedding_model = "text-embedding-3-small"
    index_path, meta_path = _write_dummy_kb(tmp_path, embedding_model)
    return index_path, meta_path, embedding_model


@pytest.fixture(scope="session")
def openai_key():
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        pytest.skip("OPENAI_API_KEY not set.")
    return key


@pytest.fixture(scope="session")
def openai_kb(tmp_path_factory, openai_key):
    root = tmp_path_factory.mktemp("kb")
    pdf_dir = root / "pdfs"
    pdf_dir.mkdir()
    _write_pdf(pdf_dir / "doc1.pdf", "Smoke test document for KB build.")

    index_path = root / "index.faiss"
    meta_path = root / "meta.pkl"
    graphrag_dir = root / "graphrag"

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

    return index_path, meta_path, graphrag_dir


@pytest.fixture
def mcp_server():
    port = _find_free_port()
    proc = Process(target=_run_server, args=(port,), daemon=True)
    proc.start()
    try:
        _wait_for_port(port)
        yield port
    finally:
        proc.terminate()
        proc.join(timeout=2)
        if proc.is_alive():
            proc.kill()
