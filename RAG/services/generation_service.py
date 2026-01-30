import os

from openai import OpenAI


def _resolve_api_key(api_key: str | None) -> str | None:
    return api_key or os.environ.get("OPENAI_API_KEY")


def _build_prompt(query: str, context: str) -> str:
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
