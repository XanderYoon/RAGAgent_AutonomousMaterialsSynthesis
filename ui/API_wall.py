import os
import streamlit as st
from openai import OpenAI


def _set_openai_key():
    api_key = (st.session_state.openai_key_input or "").strip()
    if api_key:
        st.session_state.api_key = api_key
        st.session_state.client = OpenAI(api_key=api_key)
        os.environ["OPENAI_API_KEY"] = api_key
    else:
        st.session_state.api_key = ""
        st.session_state.client = None
        os.environ.pop("OPENAI_API_KEY", None)


def API_entry():
    if "api_key" not in st.session_state:
        st.session_state.api_key = ""
    if "client" not in st.session_state:
        st.session_state.client = None

    st.header("🔓 OpenAI Key")
    st.text_input(
        "🔑 Enter your OpenAI API key",
        type="password",
        value=st.session_state.api_key,
        key="openai_key_input",
        on_change=_set_openai_key,
    )

    if st.session_state.api_key:
        st.success("✅ API key set for this session.")
    else:
        st.info("Add a key to enable OpenAI-backed features.")
