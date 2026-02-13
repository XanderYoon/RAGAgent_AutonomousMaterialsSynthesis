import os

from openai import OpenAI
from langchain_openai import OpenAIEmbeddings

from state.config import ENC
from RAG.mcp.schemas import (
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
from RAG.services.kb_service import append_kb, build_kb, load_kb
from RAG.services.retrieval_service import retrieve_context
from RAG.services.generation_service import generate_answer


def _resolve_api_key(api_key: str | None) -> str | None:
    return api_key or os.environ.get("OPENAI_API_KEY")


def _require_api_key(api_key: str | None) -> str:
    key = _resolve_api_key(api_key)
    if not key:
        raise ValueError("OpenAI API key is required.")
    os.environ["OPENAI_API_KEY"] = key
    return key


def _make_client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key)


def _make_embeddings(embedding_model: str, api_key: str) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=embedding_model, api_key=api_key)


def register_tools(mcp):
    def kb_build(params: KBBuildInput) -> KBBuildResult:
        """Build a FAISS knowledge base from PDFs and optionally run GraphRAG."""
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
        """Validate and inspect an existing knowledge base on disk."""
        _require_api_key(params.api_key)
        result = load_kb(
            index_path=params.index_path,
            meta_path=params.meta_path,
            graphrag_dir=params.graphrag_dir,
        )
        return KBLoadResult(**result)

    def kb_append(params: KBAppendInput) -> KBAppendResult:
        """Append new PDFs to an existing knowledge base on disk."""
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
        """Retrieve context and source references from a knowledge base."""
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
        """Generate an answer given a query and context text."""
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
        """Retrieve context and generate an answer in one call."""
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
