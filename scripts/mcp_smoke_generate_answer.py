import json
import os
import sys
from pathlib import Path

from fastmcp import Client

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _require_api_key():
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise SystemExit("OPENAI_API_KEY is required for this smoke test.")
    return key


def _extract_json_text(result):
    if isinstance(result, list) and result:
        content = result[0]
        text = getattr(content, "text", None)
        if text:
            return json.loads(text)
    return result


def main():
    api_key = _require_api_key()
    client = Client("http://127.0.0.1:8000/mcp/")
    query = "What does the context say about the system?"
    context_text = (
        "[dummy.txt | chunk 0]: This system builds a knowledge base from PDFs "
        "and retrieves context for answering scientific questions."
    )
    payload = {
        "params": {
            "query": query,
            "context_text": context_text,
            "api_key": api_key,
            "system": "You are a helpful scientific assistant.",
            "model": os.environ.get("QA_MODEL", "gpt-4o-mini"),
        }
    }

    import asyncio

    async def _run():
        async with client:
            return await client.call_tool("generate_answer_tool", payload)

    result = asyncio.run(_run())
    parsed = _extract_json_text(result)
    print("generate_answer_tool result:", parsed)


if __name__ == "__main__":
    main()
