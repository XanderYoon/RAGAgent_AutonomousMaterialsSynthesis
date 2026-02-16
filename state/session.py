# state/session.py
import streamlit as st
from state.config import ENC


def init_session():
    """
    Initialize all Streamlit session_state keys used by the app.
    Safe to call multiple times.
    """

    defaults = {
        # API / client
        "api_verified": False,
        "client": None,
        "api_key": None,
        "api_key_error": None,

        # Models
        "embedding_model": "text-embedding-3-small",
        "gpt_model": None,
        "temperature": 0.3,
        "diversity": 0.7,
        "enc": ENC,

        # Knowledge base
        "dimension": None,
        "index": None,
        "metadata": None,
        "db": None,
        "bm25": None,
        "all_documents": None,
        "graphrag": None,
        "index_path": None,
        "meta_path": None,

        # Uploads
        "upload_db": None,
        "upload_meta": [],
        "upload_images": [],

        # Q&A
        "last_query": "",
        "last_answer": "",
        "context_meta": "",
        "use_graphrag": False,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
