import os
import streamlit as st
from openai import OpenAI


class OpenAIKeyManager:
    ENV_KEY = "OPENAI_API_KEY"

    def __init__(self, session):
        self.s = session
        self._init_state()

    def _init_state(self):
        self.s.setdefault("api_key", "")
        self.s.setdefault("client", None)
        self.s.setdefault("api_verified", False)
        self.s.setdefault("api_key_error", None)
        self.s.setdefault("openai_key_input", self.s.api_key)

    def clear(self):
        self.s.update(
            api_key="",
            client=None,
            api_verified=False,
            api_key_error=None,
        )
        os.environ.pop(self.ENV_KEY, None)

    def set_key(self, raw_key: str):
        """Validate and set API key for this session."""
        key = (raw_key or "").strip()
        if not key:
            self.clear()
            return

        try:
            client = OpenAI(api_key=key)
            client.models.list()  # auth check

            self.s.update(
                api_key=key,
                client=client,
                api_verified=True,
                api_key_error=None,
            )
            os.environ[self.ENV_KEY] = key

        except Exception as e:
            self.clear()
            self.s.api_key_error = str(e)


def API_entry():
    """
    Streamlit UI for entering and validating an OpenAI API key.
    """

    km = OpenAIKeyManager(st.session_state)

    st.header("🔓 OpenAI Key")
    with st.form("openai_key_form"):
        st.text_input(
            "🔑 Enter your OpenAI API key",
            type="password",
            key="openai_key_input",
        )
        if st.form_submit_button("Submit"):
            km.set_key(st.session_state.openai_key_input)

    if st.session_state.api_verified:
        st.success("✅ API key set for this session.")
    elif st.session_state.api_key_error:
        st.error("❌ Invalid API key")
        print("OpenAI key auth error:\n", st.session_state.api_key_error)
    else:
        st.info("Add a key to enable OpenAI-backed features.")
