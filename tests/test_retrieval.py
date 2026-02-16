import pytest

from RAG.retrieval import RetrievalError, retrieve_context


def test_retrieve_context_requires_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(RetrievalError, match="API key"):
        retrieve_context(
            query="What is this?",
            index_path="missing.index",
            meta_path="missing.pkl",
            graphrag_dir=None,
            api_key=None,
        )
