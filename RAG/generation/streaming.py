import time

import streamlit as st

from config import STREAM_DELAY


def stream_answer(client, query: str, context: str) -> str:
    """Stream an answer to the UI and return it."""
    system = (
        "You are an expert scientific research assistant. Use the context provided from research papers to answer "
        "the user query as accurately as possible. Provide detailed responses using as much of the provided context "
        "as possible. If the answer is not clearly found in the context, respond with: "
        "'The context does not provide enough information to answer this question.' and default to your parametric knowledge to give a response "
        "If the provided context is not enough to answer the user, use web search to find relevant information. "
        "At the end of your answer, also mention the source file names or uploaded image names referenced in the context "
        "(e.g., [DL-rheed-harris-SI.pdf | chunk 1]) or (e.g., [uploaded/image.png | image])."
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
