from typing import List

from pydantic import BaseModel, Field


class KBBuildInput(BaseModel):
    pdf_dir: str
    index_path: str
    meta_path: str
    graphrag_dir: str
    embedding_model: str = "text-embedding-3-small"
    api_key: str
    run_graphrag: bool


class KBBuildResult(BaseModel):
    total_chunks: int
    total_tokens: int
    estimated_cost: float
    warnings: List[str] = Field(default_factory=list)


class KBLoadInput(BaseModel):
    index_path: str
    meta_path: str
    graphrag_dir: str
    api_key: str


class KBLoadResult(BaseModel):
    status: str
    embedding_model: str = "text-embedding-3-small"
    dimension: int
    chunk_count: int
    graphrag_dir: str = ""


class KBAppendInput(BaseModel):
    index_path: str
    meta_path: str
    append_folder: str
    graphrag_dir: str
    api_key: str
    run_graphrag: bool


class KBAppendResult(BaseModel):
    new_chunks: int
    new_tokens: int
    estimated_cost: float
    warnings: List[str] = Field(default_factory=list)


class RetrieveContextInput(BaseModel):
    query: str
    index_path: str
    meta_path: str
    graphrag_dir: str
    api_key: str
    diversity: float = Field(ge=0.0, le=1.0)
    top_k_faiss: int = 5
    use_graphrag: bool = False
    retriever_model: str = "gpt-4o-mini"
    compressor_model: str = "gpt-4o-mini"

class SourceRef(BaseModel):
    source: str
    chunk_id: int


class RetrieveContextResult(BaseModel):
    context_text: str
    sources: List[SourceRef] = Field(default_factory=list)
    embedding_model: str = "text-embedding-3-small"


class GenerateAnswerInput(BaseModel):
    query: str
    context_text: str
    model: str
    system: str
    api_key: str
    enable_web_search: bool


class GenerateAnswerResult(BaseModel):
    answer: str


class RetrieveAndAnswerInput(BaseModel):
    query: str
    index_path: str
    meta_path: str
    graphrag_dir: str
    api_key: str
    diversity: float = Field(ge=0.0, le=1.0)
    top_k_faiss: int = 5
    use_graphrag: bool = False
    retriever_model: str = "gpt-4o-mini"
    compressor_model: str = "gpt-4o-mini"
    response_model: str = "gpt-3.5-turbo"
    system: str
    enable_web_search: bool


class RetrieveAndAnswerResult(BaseModel):
    answer: str
    context_text: str
    sources: List[SourceRef] = Field(default_factory=list)
