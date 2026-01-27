# app.py
import streamlit as st

from state.session import init_session

from ui.API_wall import API_entry
from ui.kb_setup import kb_setup
from ui.qa_panel import qa_panel

from langchain_openai import OpenAIEmbeddings

init_session()

# --------------------------------
# App layout
# --------------------------------
st.markdown("## 📄 RAG Agent — Autonomous Synthesis")
st.divider()

# --------------------------------
# API entry / auth
# --------------------------------
API_entry()
client = st.session_state.client
st.divider()

# --------------------------------
# Knowledge base setup
# --------------------------------
embeddings_obj = None
if st.session_state.api_verified:
    embeddings_obj = OpenAIEmbeddings(
        model=st.session_state.get("embedding_model", "text-embedding-3-small"),
        api_key=st.session_state.api_key,
    )

kb_setup(
    client=client,
    embeddings_obj=embeddings_obj,
)

st.divider()

# --------------------------------
# Q&A
# --------------------------------
qa_panel(client=client)
