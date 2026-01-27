import streamlit as st
from langchain.retrievers import EnsembleRetriever, MultiQueryRetriever
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_community.document_transformers import LongContextReorder
from langchain_openai import ChatOpenAI

from config import TOP_K_TEXT_FAISS


class QAContextRetriever:
    """Hybrid retriever with LLM compression."""

    def retrieve(self, query: str) -> str:
        vs = st.session_state.db.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": TOP_K_TEXT_FAISS,
                "fetch_k": TOP_K_TEXT_FAISS * 2,
                "lambda_mult": 1.0 - st.session_state.diversity,
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

        return "\n\n".join(
            f"[{d.metadata['source']} | chunk {d.metadata['chunk_id']}]: {d.page_content}"
            for d in docs
        )
