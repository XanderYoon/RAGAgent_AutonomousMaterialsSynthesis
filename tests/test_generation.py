import pytest

from RAG.generation import GenerationError, generate_answer


def test_generate_answer_requires_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(GenerationError, match="API key"):
        generate_answer(
            query="What is this?",
            context_text="[doc | chunk 0]: content",
            model="gpt-4o-mini",
            system="You are a test assistant.",
            api_key=None,
            enable_web_search=False,
        )
