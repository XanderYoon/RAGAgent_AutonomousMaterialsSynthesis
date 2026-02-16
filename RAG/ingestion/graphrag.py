# RAG/ingestion/graphrag.py
from pathlib import Path
import subprocess

from langchain_core.retrievers import BaseRetriever


# ----------------------------------------
# Run GraphRAG indexing pipeline
# ----------------------------------------
def run_graphrag_cli(root_dir: str, input_dir: str, output_dir: str, api_key: str):
    """Run GraphRAG initialization and indexing commands. """
    settings_path = Path(root_dir) / "settings.yaml"
    env_path = Path(root_dir) / ".env"

    with open(env_path, "w") as f:
        f.write(f"GRAPHRAG_API_KEY={api_key}\n")

    if not settings_path.exists():
        subprocess.run(
            ["graphrag", "init", "--root", str(root_dir)],
            check=True,
        )

    result = subprocess.run(
        ["graphrag", "index", "--root", str(root_dir)],
        capture_output=True,
        text=True,
    )

    return result.returncode == 0


# ----------------------------------------
# Static retriever wrapper (LangChain)
# ----------------------------------------
class StaticGraphRetriever(BaseRetriever):
    def __init__(self, graph_docs):
        super().__init__()
        self.graph_docs = graph_docs

    def _get_relevant_documents(self, query, *, run_manager=None):
        """Return precomputed graph documents for synchronous retrieval. """
        return self.graph_docs

    async def _aget_relevant_documents(self, query, *, run_manager=None):
        """Return precomputed graph documents for asynchronous retrieval."""
        return self.graph_docs
