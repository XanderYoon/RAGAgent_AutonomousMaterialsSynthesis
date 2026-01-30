import os
import pickle
from pathlib import Path

import faiss
from langchain.retrievers import EnsembleRetriever, MultiQueryRetriever
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_community.document_transformers import LongContextReorder
from langchain_community.retrievers import BM25Retriever
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from state.config import TOP_K_TEXT_BM25, TOP_K_TEXT_FAISS
from RAG.retrieval.graphrag_query import query_graphrag, GraphRAGQueryError


def _resolve_api_key(api_key: str | None) -> str | None:
    return api_key or os.environ.get("OPENAI_API_KEY")


def _load_index_and_metadata(index_path: str, meta_path: str):
    index = faiss.read_index(index_path)
    with open(meta_path, "rb") as f:
        metadata = pickle.load(f)
    if not metadata:
        raise ValueError("Metadata file is empty.")
    return index, metadata


def _build_docs(metadata):
    from langchain_core.documents import Document

    docs = []
    for meta in metadata:
        docs.append(
            Document(
                page_content=meta["text"],
                metadata={
                    "source": meta["source"],
                    "chunk_id": meta["chunk_id"],
                    "original_content": meta["text"],
                },
            )
        )
    return docs


def _build_faiss_retriever(
    index,
    metadata,
    embeddings,
    top_k_faiss: int,
    diversity: float,
    docs,
):
    from langchain_community.vectorstores import FAISS
    from langchain_community.docstore.in_memory import InMemoryDocstore

    ids = [str(i) for i in range(len(metadata))]
    docs_dict = {ids[i]: docs[i] for i in range(len(metadata))}
    docstore = InMemoryDocstore(docs_dict)
    index_to_docstore_id = {i: ids[i] for i in range(len(ids))}

    db = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id,
    )
    return db.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": top_k_faiss,
            "fetch_k": top_k_faiss * 2,
            "lambda_mult": 1.0 - diversity,
        },
    )


def retrieve_context(
    *,
    query: str,
    index_path: str,
    meta_path: str,
    graphrag_dir: str | None,
    api_key: str | None = None,
    diversity: float = 0.3,
    top_k_faiss: int | None = None,
    use_graphrag: bool = True,
    retriever_model: str = "gpt-4o-mini",
    compressor_model: str = "gpt-4o-mini",
):
    key = _resolve_api_key(api_key)
    if not key:
        raise ValueError("OpenAI API key is required for retrieval.")

    index, metadata = _load_index_and_metadata(index_path, meta_path)
    embedding_model = metadata[0]["embedding_model"]
    embeddings = OpenAIEmbeddings(model=embedding_model, api_key=key)

    top_k = top_k_faiss or TOP_K_TEXT_FAISS
    docs = _build_docs(metadata)
    vs = _build_faiss_retriever(
        index, metadata, embeddings, top_k, diversity, docs
    )

    bm25 = BM25Retriever.from_documents(docs, k=TOP_K_TEXT_BM25)

    hybrid = EnsembleRetriever(
        retrievers=[vs, bm25],
        weights=[0.5, 0.5],
    )

    mq = MultiQueryRetriever.from_llm(
        retriever=hybrid,
        llm=ChatOpenAI(model=retriever_model, api_key=key),
        include_original=True,
    )

    compressor = LLMChainExtractor.from_llm(
        ChatOpenAI(model=compressor_model, temperature=0, api_key=key)
    )

    retriever = ContextualCompressionRetriever(
        base_retriever=mq,
        base_compressor=compressor,
    )

    docs = retriever.invoke(query)
    docs = LongContextReorder().transform_documents(docs)

    context_lines = [
        f"[{d.metadata['source']} | chunk {d.metadata['chunk_id']}]: {d.page_content}"
        for d in docs
    ]

    sources = [
        {"source": d.metadata["source"], "chunk_id": d.metadata["chunk_id"]}
        for d in docs
    ]

    if use_graphrag and graphrag_dir:
        root = Path(graphrag_dir)
        if root.is_dir():
            try:
                graph_text = query_graphrag(root, query)
                if graph_text:
                    context_lines.append("\n[GraphRAG Context]:\n" + graph_text)
                    sources.append({"source": "graphrag", "chunk_id": -1})
            except GraphRAGQueryError:
                pass

    return {
        "context_text": "\n\n".join(context_lines),
        "sources": sources,
        "embedding_model": embedding_model,
    }
