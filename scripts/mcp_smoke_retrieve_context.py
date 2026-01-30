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
    index_path = os.environ.get("KB_INDEX_PATH")
    meta_path = os.environ.get("KB_META_PATH")
    if not index_path or not meta_path:
        raise SystemExit(
            "Set KB_INDEX_PATH and KB_META_PATH to an existing knowledge base."
        )

    client = Client("http://127.0.0.1:8000/mcp/")
    query = "What is the main topic of this knowledge base?"
    payload = {
        "params": {
            "query": query,
            "index_path": index_path,
            "meta_path": meta_path,
            "graphrag_dir": os.environ.get("KB_GRAPHRAG_DIR"),
            "api_key": api_key,
        }
    }

    import asyncio

    async def _run():
        async with client:
            return await client.call_tool("retrieve_context_tool", payload)

    result = asyncio.run(_run())
    parsed = _extract_json_text(result)
    print("retrieve_context_tool result:", parsed)


if __name__ == "__main__":
    main()
