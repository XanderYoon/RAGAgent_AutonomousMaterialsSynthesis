import os

from openai import OpenAI


class GenerationError(Exception):
    """Raised when generation pipeline execution fails."""


def _resolve_api_key(api_key: str | None) -> str | None:
    """Resolve API key from explicit argument or environment."""
    return api_key or os.environ.get("OPENAI_API_KEY")


def _build_prompt(query: str, context: str) -> str:
    """Create generation prompt from query and retrieved context."""
    return f"""
User query: {query}

--- BEGIN CONTEXT ---
{context}
--- END CONTEXT ---

Answer:
""".strip()


def generate_answer(
    *,
    query: str,
    context_text: str,
    model: str,
    system: str,
    api_key: str | None = None,
    enable_web_search: bool = False,
):
    """Generate a final answer from context using the Responses API."""
    key = _resolve_api_key(api_key)
    if not key:
        raise GenerationError("OpenAI API key is required for generation.")

    client = OpenAI(api_key=key)
    prompt = _build_prompt(query, context_text)

    kwargs = {"model": model, "instructions": system, "input": prompt}
    if enable_web_search:
        kwargs["tools"] = [{"type": "web_search_preview"}]

    try:
        response = client.responses.create(**kwargs)
    except Exception as exc:
        raise GenerationError("Failed to generate answer from model response.") from exc

    return response.output_text
