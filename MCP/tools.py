import os

from openai import OpenAI
from langchain_openai import OpenAIEmbeddings

from state.config import ENC
from MCP.schemas import (
    KBAppendInput,
    KBAppendResult,
    KBBuildInput,
    KBBuildResult,
    KBLoadInput,
    KBLoadResult,
    GenerateAnswerInput,
    GenerateAnswerResult,
    RetrieveAndAnswerInput,
    RetrieveAndAnswerResult,
    RetrieveContextInput,
    RetrieveContextResult,
)
from RAG.kb_builder import append_kb, build_kb, load_kb
from RAG.retrieval import retrieve_context
from RAG.generation import generate_answer


def _resolve_api_key(api_key: str | None) -> str | None:
    """Resolve API key from argument or environment."""
    return api_key or os.environ.get("OPENAI_API_KEY")


def _require_api_key(api_key: str | None) -> str:
    """Validate and persist API key for downstream SDK calls."""
    key = _resolve_api_key(api_key)
    if not key:
        raise ValueError("OpenAI API key is required.")
    os.environ["OPENAI_API_KEY"] = key
    return key


def _make_client(api_key: str) -> OpenAI:
    """Create an OpenAI client with a fixed API key."""
    return OpenAI(api_key=api_key)


def _make_embeddings(embedding_model: str, api_key: str) -> OpenAIEmbeddings:
    """Create an embeddings wrapper for the requested model."""
    return OpenAIEmbeddings(model=embedding_model, api_key=api_key)


def register_tools(mcp):
    """Register MCP tool handlers for KB and QA workflows. """
    def kb_build(params: KBBuildInput) -> KBBuildResult:
        """Build a FAISS knowledge base from PDFs and optionally run GraphRAG.

        Args:
            params: Validated KB build input payload.

        Returns:
            Structured build result with counts, cost, and warnings.
        """
        key = _require_api_key(params.api_key)
        client = _make_client(key)
        embeddings = _make_embeddings(params.embedding_model, key)
        result = build_kb(
            client=client,
            embeddings=embeddings,
            enc=ENC,
            pdf_dir=params.pdf_dir,
            index_path=params.index_path,
            meta_path=params.meta_path,
            graphrag_dir=params.graphrag_dir,
            embedding_model=params.embedding_model,
            run_graphrag=params.run_graphrag,
            api_key=key,
        )
        return KBBuildResult(**result)

    def kb_load(params: KBLoadInput) -> KBLoadResult:
        """Validate and inspect an existing knowledge base on disk.

        Args:
            params: Validated KB load input payload.

        Returns:
            Structured load result with model and dimension details.
        """
        _require_api_key(params.api_key)
        result = load_kb(
            index_path=params.index_path,
            meta_path=params.meta_path,
            graphrag_dir=params.graphrag_dir,
        )
        return KBLoadResult(**result)

    def kb_append(params: KBAppendInput) -> KBAppendResult:
        """Append new PDFs to an existing knowledge base on disk.

        Args:
            params: Validated KB append input payload.

        Returns:
            Structured append result with counts, cost, and warnings.
        """
        key = _require_api_key(params.api_key)
        client = _make_client(key)
        result = append_kb(
            client=client,
            enc=ENC,
            index_path=params.index_path,
            meta_path=params.meta_path,
            append_folder=params.append_folder,
            graphrag_dir=params.graphrag_dir,
            run_graphrag=params.run_graphrag,
            api_key=key,
        )
        return KBAppendResult(**result)

    def retrieve_context_tool(params: RetrieveContextInput) -> RetrieveContextResult:
        """Retrieve context and source references from a knowledge base.

        Args:
            params: Validated retrieval input payload.

        Returns:
            Retrieved context text, source references, and embedding model name.
        """
        key = _require_api_key(params.api_key)
        result = retrieve_context(
            query=params.query,
            index_path=params.index_path,
            meta_path=params.meta_path,
            graphrag_dir=params.graphrag_dir,
            api_key=key,
            diversity=params.diversity,
            top_k_faiss=params.top_k_faiss,
            use_graphrag=params.use_graphrag,
            retriever_model=params.retriever_model,
            compressor_model=params.compressor_model,
        )
        return RetrieveContextResult(**result)

    def generate_answer_tool(params: GenerateAnswerInput) -> GenerateAnswerResult:
        """Generate an answer from query and context text.

        Args:
            params: Validated generation input payload.

        Returns:
            Structured generation result containing answer text.
        """
        key = _require_api_key(params.api_key)
        answer = generate_answer(
            query=params.query,
            context_text=params.context_text,
            model=params.model,
            system=params.system,
            api_key=key,
            enable_web_search=params.enable_web_search,
        )
        return GenerateAnswerResult(answer=answer)

    def retrieve_and_answer_tool(
        params: RetrieveAndAnswerInput,
    ) -> RetrieveAndAnswerResult:
        """Retrieve context and generate an answer in one call.

        Args:
            params: Validated retrieve-and-answer input payload.

        Returns:
            Combined response containing answer, context text, and sources.
        """
        key = _require_api_key(params.api_key)
        context = retrieve_context(
            query=params.query,
            index_path=params.index_path,
            meta_path=params.meta_path,
            graphrag_dir=params.graphrag_dir,
            api_key=key,
            diversity=params.diversity,
            top_k_faiss=params.top_k_faiss,
            use_graphrag=params.use_graphrag,
            retriever_model=params.retriever_model,
            compressor_model=params.compressor_model,
        )
        answer = generate_answer(
            query=params.query,
            context_text=context["context_text"],
            model=params.response_model,
            system=params.system,
            api_key=key,
            enable_web_search=params.enable_web_search,
        )
        return RetrieveAndAnswerResult(
            answer=answer,
            context_text=context["context_text"],
            sources=context["sources"],
        )

    mcp.tool()(kb_build)
    mcp.tool()(kb_load)
    mcp.tool()(kb_append)
    mcp.tool()(retrieve_context_tool)
    mcp.tool()(generate_answer_tool)
    mcp.tool()(retrieve_and_answer_tool)
