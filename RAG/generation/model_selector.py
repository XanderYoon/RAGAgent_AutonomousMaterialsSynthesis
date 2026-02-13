import streamlit as st

DEFAULT_MODEL_OPTIONS = [
    "gpt-4.1-2025-04-14",
    "gpt-4o-2024-08-06",
    "gpt-5",
    "gpt-5-thinking",
    "gpt-5-pro",
    "o4-mini-2025-04-16",
    "o4-mini-deep-research-2025-06-26",
]


def _is_allowed_model(model_id: str) -> bool:
    """Check whether a model ID is allowed in the selector."""
    mid = (model_id or "").lower()
    return mid.startswith("o") or (mid.startswith("gpt-") and "codex" not in mid)


def load_model_options(client):
    """Load and cache selectable model IDs."""
    cached = st.session_state.get("available_models")
    if cached:
        return cached

    if client is None:
        return DEFAULT_MODEL_OPTIONS

    try:
        data = client.models.list()
        models = [m.id for m in getattr(data, "data", [])]
        options = sorted({m for m in models if _is_allowed_model(m)})
        if options:
            st.session_state.available_models = options
            return options
    except Exception:
        pass

    return DEFAULT_MODEL_OPTIONS
