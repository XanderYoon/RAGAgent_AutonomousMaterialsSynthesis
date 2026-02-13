import asyncio
import json

import pytest
from fastmcp import Client


def _extract_json_text(result):
    if isinstance(result, list) and result:
        content = result[0]
        text = getattr(content, "text", None)
        if text:
            return json.loads(text)
    return result


def test_mcp_kb_load_http(dummy_kb, mcp_server):
    index_path, meta_path, _ = dummy_kb
    client = Client(f"http://127.0.0.1:{mcp_server}/mcp/")

    async def _run():
        async with client:
            return await client.call_tool(
                "kb_load",
                {
                    "params": {
                        "index_path": str(index_path),
                        "meta_path": str(meta_path),
                        "graphrag_dir": "",
                        "api_key": "dummy-api-key",
                    }
                },
            )

    result = asyncio.run(_run())
    parsed = _extract_json_text(result)
    assert parsed["status"] == "ok"


@pytest.mark.integration
@pytest.mark.requires_openai
def test_mcp_generate_answer_tool(openai_key, mcp_server):
    client = Client(f"http://127.0.0.1:{mcp_server}/mcp/")
    payload = {
        "params": {
            "query": "What does the context say about the system?",
            "context_text": (
                "[dummy.txt | chunk 0]: This system builds a knowledge base "
                "from PDFs and retrieves context for answering scientific "
                "questions."
            ),
            "api_key": openai_key,
            "system": "You are a helpful scientific assistant.",
            "model": "gpt-4o-mini",
            "enable_web_search": False,
        }
    }

    async def _run():
        async with client:
            return await client.call_tool("generate_answer_tool", payload)

    result = asyncio.run(_run())
    parsed = _extract_json_text(result)
    assert isinstance(parsed.get("answer"), str)
    assert parsed["answer"].strip()


@pytest.mark.integration
@pytest.mark.requires_openai
def test_mcp_retrieve_context_tool(openai_key, openai_kb, mcp_server):
    index_path, meta_path, graphrag_dir = openai_kb
    client = Client(f"http://127.0.0.1:{mcp_server}/mcp/")
    payload = {
        "params": {
            "query": "What is the main topic of this knowledge base?",
            "index_path": str(index_path),
            "meta_path": str(meta_path),
            "graphrag_dir": str(graphrag_dir),
            "api_key": openai_key,
            "diversity": 0.3,
            "top_k_faiss": 8,
            "use_graphrag": False,
            "retriever_model": "gpt-4o-mini",
            "compressor_model": "gpt-4o-mini",
        }
    }

    async def _run():
        async with client:
            return await client.call_tool("retrieve_context_tool", payload)

    result = asyncio.run(_run())
    parsed = _extract_json_text(result)
    assert isinstance(parsed.get("context_text"), str)
    assert parsed["context_text"].strip()


@pytest.mark.integration
@pytest.mark.requires_openai
def test_mcp_retrieve_and_answer_tool(openai_key, openai_kb, mcp_server):
    index_path, meta_path, graphrag_dir = openai_kb
    client = Client(f"http://127.0.0.1:{mcp_server}/mcp/")
    payload = {
        "params": {
            "query": "Summarize the key information available.",
            "index_path": str(index_path),
            "meta_path": str(meta_path),
            "graphrag_dir": str(graphrag_dir),
            "api_key": openai_key,
            "diversity": 0.3,
            "top_k_faiss": 8,
            "use_graphrag": False,
            "retriever_model": "gpt-4o-mini",
            "compressor_model": "gpt-4o-mini",
            "system": (
                "You are a helpful scientific assistant. Use the provided "
                "context and cite sources inline with [source | chunk id]."
            ),
            "model": "gpt-4o-mini",
            "enable_web_search": False,
        }
    }

    async def _run():
        async with client:
            return await client.call_tool("retrieve_and_answer_tool", payload)

    result = asyncio.run(_run())
    parsed = _extract_json_text(result)
    assert isinstance(parsed.get("answer"), str)
    assert parsed["answer"].strip()
