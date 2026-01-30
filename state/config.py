from langchain.globals import set_llm_cache
from langchain_community.cache import SQLiteCache
import tiktoken

set_llm_cache(SQLiteCache(database_path=".cache.db"))

# Embedding dimensions
EMBEDDING_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072
}

TOKENS_PER_CHUNK = 300
WORDS_PER_CHUNK_OVERLAP = int(TOKENS_PER_CHUNK / 5)  # ~20%

# Retrieval sizes
TOP_K_TEXT_FAISS = 50
TOP_K_TEXT_BM25 = 50
TOP_K_UPLOAD_FAISS = 25
TOP_K_GRAPH = 25

STREAM_DELAY = 0.08

TOKENIZER_NAME = "cl100k_base"
ENC = tiktoken.get_encoding(TOKENIZER_NAME)