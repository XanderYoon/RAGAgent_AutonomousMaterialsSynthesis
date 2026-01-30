import os
import streamlit as st
from langchain.retrievers import EnsembleRetriever, MultiQueryRetriever
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_community.document_transformers import LongContextReorder
from langchain_openai import ChatOpenAI

from state.config import TOP_K_TEXT_FAISS, TOP_K_UPLOAD_FAISS
from state.schemas import RetrievalRequest
from RAG.retrieval.graphrag_query import query_graphrag, GraphRAGQueryError


def _get_cb(callbacks, name):
    if callbacks is None:
        return None
    return getattr(callbacks, name, None)


def _call(cb, *args, **kwargs):
    if cb:
        cb(*args, **kwargs)


class QAContextRetriever:
    """Hybrid retriever with LLM compression."""

    def retrieve(
        self,
        query: str,
        use_uploads: bool = True,
        use_graphrag: bool = True,
        callbacks=None,
    ) -> str:
        req = RetrievalRequest(
            query=query,
            use_uploads=use_uploads,
            use_graphrag=use_graphrag,
            top_k_faiss=TOP_K_TEXT_FAISS,
            top_k_uploads=TOP_K_UPLOAD_FAISS,
            diversity=st.session_state.diversity,
        )
        vs = st.session_state.db.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": req.top_k_faiss,
                "fetch_k": req.top_k_faiss * 2,
                "lambda_mult": 1.0 - req.diversity,
            },
        )

        hybrid = EnsembleRetriever(
            retrievers=[vs, st.session_state.bm25],
            weights=[0.5, 0.5],
        )

        mq = MultiQueryRetriever.from_llm(
            retriever=hybrid,
            llm=ChatOpenAI(model="gpt-4o-mini"),
            include_original=True,
        )

        compressor = LLMChainExtractor.from_llm(
            ChatOpenAI(model="gpt-4o-mini", temperature=0)
        )

        retriever = ContextualCompressionRetriever(
            base_retriever=mq,
            base_compressor=compressor,
        )

        docs = retriever.invoke(query)
        docs = LongContextReorder().transform_documents(docs)

        upload_docs = []
        if req.use_uploads and st.session_state.get("upload_db") is not None:
            vs_upload = st.session_state.upload_db.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": req.top_k_uploads,
                    "fetch_k": req.top_k_uploads * 2,
                    "lambda_mult": 1.0 - req.diversity,
                },
            )
            mq_upload = MultiQueryRetriever.from_llm(
                retriever=vs_upload,
                llm=ChatOpenAI(model="gpt-4o-mini"),
                include_original=True,
            )
            retriever_upload = ContextualCompressionRetriever(
                base_retriever=mq_upload,
                base_compressor=compressor,
            )
            upload_docs = retriever_upload.invoke(query)
            upload_docs = LongContextReorder().transform_documents(upload_docs)

        merged_docs = docs
        if upload_docs:
            merged_docs.extend(upload_docs)

        context_lines = [
            f"[{d.metadata['source']} | chunk {d.metadata['chunk_id']}]: {d.page_content}"
            for d in merged_docs
        ]

        if req.use_uploads and st.session_state.get("upload_images"):
            for img in st.session_state.upload_images:
                context_lines.append(
                    f"[uploaded/{img['name']} | image]: (image attached)"
                )

        if req.use_graphrag:
            graph_root = st.session_state.get("graphrag")
            if graph_root and os.path.isdir(graph_root):
                try:
                    graph_text = query_graphrag(graph_root, query)
                    if graph_text:
                        context_lines.append(
                            "\n[GraphRAG Context]:\n" + graph_text
                        )
                    else:
                        _call(
                            _get_cb(callbacks, "info"),
                            "ℹ️ No Knowledge-Graph context found for this query.",
                        )
                except GraphRAGQueryError as exc:
                    _call(
                        _get_cb(callbacks, "warning"),
                        f"⚠️ Error while retrieving from Knowledge-Graph: {exc}",
                    )
            else:
                _call(
                    _get_cb(callbacks, "info"),
                    "ℹ️ No Knowledge-Graph directory loaded - skipping graph-based retrieval.",
                )

        return "\n\n".join(context_lines)
