from typing import List, Optional

from pydantic import BaseModel, Field


class KBBuildInput(BaseModel):
    pdf_dir: str
    index_path: str
    meta_path: str
    graphrag_dir: Optional[str] = None
    embedding_model: str = "text-embedding-3-small"
    api_key: Optional[str] = None
    run_graphrag: bool = True


class KBBuildResult(BaseModel):
    total_chunks: int
    total_tokens: int
    estimated_cost: float
    warnings: List[str] = Field(default_factory=list)


class KBLoadInput(BaseModel):
    index_path: str
    meta_path: str
    graphrag_dir: Optional[str] = None
    embedding_model: Optional[str] = None


class KBLoadResult(BaseModel):
    status: str
    embedding_model: str
    dimension: int
    chunk_count: int
    graphrag_dir: str = ""


class KBAppendInput(BaseModel):
    index_path: str
    meta_path: str
    append_folder: str
    graphrag_dir: Optional[str] = None
    api_key: Optional[str] = None
    run_graphrag: bool = True


class KBAppendResult(BaseModel):
    new_chunks: int
    new_tokens: int
    estimated_cost: float
    warnings: List[str] = Field(default_factory=list)


class RetrieveContextInput(BaseModel):
    query: str
    index_path: str
    meta_path: str
    graphrag_dir: Optional[str] = None
    api_key: Optional[str] = None
    diversity: float = 0.3
    top_k_faiss: Optional[int] = None
    use_graphrag: bool = True
    retriever_model: str = "gpt-4o-mini"
    compressor_model: str = "gpt-4o-mini"


class SourceRef(BaseModel):
    source: str
    chunk_id: int


class RetrieveContextResult(BaseModel):
    context_text: str
    sources: List[SourceRef] = Field(default_factory=list)
    embedding_model: str


class GenerateAnswerInput(BaseModel):
    query: str
    context_text: str
    model: str = "gpt-4o-mini"
    system: str
    api_key: Optional[str] = None
    enable_web_search: bool = False


class GenerateAnswerResult(BaseModel):
    answer: str


class RetrieveAndAnswerInput(BaseModel):
    query: str
    index_path: str
    meta_path: str
    graphrag_dir: Optional[str] = None
    api_key: Optional[str] = None
    diversity: float = 0.3
    top_k_faiss: Optional[int] = None
    use_graphrag: bool = True
    retriever_model: str = "gpt-4o-mini"
    compressor_model: str = "gpt-4o-mini"
    model: str = "gpt-4o-mini"
    system: str
    enable_web_search: bool = False


class RetrieveAndAnswerResult(BaseModel):
    answer: str
    context_text: str
    sources: List[SourceRef] = Field(default_factory=list)
