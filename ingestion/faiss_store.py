# ingestion/faiss_store.py
import faiss
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.documents import Document


# ----------------------------------------
# Build FAISS from embeddings + metadata
# ----------------------------------------
def build_faiss_from_embeddings(
    embeddings,
    metadata,
    embedding_function,
    dimension,
):
    embedding_matrix = np.array(embeddings, dtype="float32")

    index = faiss.IndexFlatL2(dimension)
    index.add(embedding_matrix)

    ids = [str(i) for i in range(len(metadata))]
    docs_dict = {
        ids[i]: Document(
            page_content=meta["text"],
            metadata={
                "source": meta["source"],
                "chunk_id": meta["chunk_id"],
                "original_content": meta["text"],
            }
        )
        for i, meta in enumerate(metadata)
    }

    docstore = InMemoryDocstore(docs_dict)
    index_to_docstore_id = {i: ids[i] for i in range(len(ids))}

    db = FAISS(
        embedding_function=embedding_function,
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id,
    )

    return index, db, list(docstore._dict.values())


# ----------------------------------------
# Rebuild FAISS wrapper from disk
# ----------------------------------------
def load_faiss_from_disk(
    index,
    metadata,
    embedding_function,
):
    ids = [str(i) for i in range(len(metadata))]
    docs_dict = {
        ids[i]: Document(
            page_content=meta["text"],
            metadata={
                "source": meta["source"],
                "chunk_id": meta["chunk_id"],
                "original_content": meta["text"],
            }
        )
        for i, meta in enumerate(metadata)
    }

    docstore = InMemoryDocstore(docs_dict)
    index_to_docstore_id = {i: ids[i] for i in range(len(ids))}

    db = FAISS(
        embedding_function=embedding_function,
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id,
    )

    return db, list(docstore._dict.values())
