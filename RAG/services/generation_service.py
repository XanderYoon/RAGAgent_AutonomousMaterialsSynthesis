import os

from openai import OpenAI


def _resolve_api_key(api_key: str | None) -> str | None:
    """Resolve API key from explicit argument or environment.

    Args:
        api_key: API key passed by caller, if any.

    Returns:
        API key string when available, otherwise ``None``.
    """
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
    """Generate a final answer from context using the Responses API.

    Args:
        query: User question text.
        context_text: Retrieved context text used as grounding content.
        model: Model identifier used for generation.
        system: System instruction string.
        api_key: Optional API key override.
        enable_web_search: Whether to enable web search tool in generation.

    Returns:
        Generated answer text from the model response.

    Raises:
        ValueError: If no API key is available for model calls.
    """
    key = _resolve_api_key(api_key)
    if not key:
        raise ValueError("OpenAI API key is required for generation.")

    client = OpenAI(api_key=key)
    prompt = _build_prompt(query, context_text)

    kwargs = {"model": model, "instructions": system, "input": prompt}
    if enable_web_search:
        kwargs["tools"] = [{"type": "web_search_preview"}]

    response = client.responses.create(**kwargs)
    return response.output_text
