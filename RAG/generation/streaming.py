import time

import streamlit as st

from config import STREAM_DELAY


def stream_answer(client, query: str, context: str) -> str:
    """Stream an answer to the UI and return it."""
    system = (
        "You are an expert scientific research assistant. "
        "Use the provided context to answer accurately."
    )

    prompt = f"""
User query: {query}

--- BEGIN CONTEXT ---
{context}
--- END CONTEXT ---

Answer:
""".strip()

    placeholder = st.empty()
    answer = ""

    with client.chat.completions.stream(
        model=st.session_state.gpt_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=st.session_state.temperature,
    ) as stream:
        for event in stream:
            if event.type == "content.delta":
                answer += event.delta
                placeholder.markdown(answer + "▌")
                time.sleep(STREAM_DELAY)
            elif event.type == "content.done":
                placeholder.markdown(answer)

    return answer
