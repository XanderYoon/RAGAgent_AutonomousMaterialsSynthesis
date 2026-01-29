# RAG/retrieval/graphrag_query.py
import subprocess
from pathlib import Path
import textwrap


class GraphRAGQueryError(RuntimeError):
    """Raised when GraphRAG query fails."""


def query_graphrag(root_dir, query, max_chars=6000):
    root = Path(root_dir)
    result = subprocess.run(
        [
            "graphrag",
            "query",
            "--root",
            str(root),
            "--method",
            "global",
            "--query",
            query,
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise GraphRAGQueryError(result.stderr.strip())

    answer = result.stdout.strip()
    if not answer:
        return None

    return textwrap.shorten(answer, width=max_chars, placeholder=" ...")
