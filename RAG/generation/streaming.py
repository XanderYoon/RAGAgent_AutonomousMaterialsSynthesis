import time

import streamlit as st

from state.config import STREAM_DELAY

_IMAGE_MODELS = {"gpt-4o-2024-08-06", "gpt-4.1-2025-04-14"}


def _build_prompt(query: str, context: str) -> str:
    return f"""
User query: {query}

--- BEGIN CONTEXT ---
{context}
--- END CONTEXT ---

Answer:
""".strip()


def stream_answer(client, query: str, context: str, images=None) -> str:
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

    model = st.session_state.gpt_model
    prompt = _build_prompt(query, context)

    placeholder = st.empty()
    answer = ""

    if model == "o4-mini-deep-research-2025-06-26":
        with client.responses.stream(
            model=model,
            instructions=system,
            tools=[{"type": "web_search_preview"}],
            input=prompt,
        ) as stream:
            for event in stream:
                if event.type == "response.output_text.delta":
                    answer += event.delta
                    placeholder.markdown(answer + "▌")
                    time.sleep(STREAM_DELAY)
                elif event.type in {"response.output_text.done", "response.completed"}:
                    placeholder.markdown(answer)
        return answer

    if model in {"gpt-5", "gpt-5-thinking", "gpt-5-pro"}:
        model_id = "gpt-5" if model == "gpt-5-thinking" else model
        kwargs = {"model": model_id, "instructions": system, "input": prompt}
        if model == "gpt-5-thinking":
            kwargs["reasoning"] = {"effort": "high"}
        try:
            with client.responses.stream(**kwargs) as stream:
                for event in stream:
                    if event.type == "response.output_text.delta":
                        answer += event.delta
                        placeholder.markdown(answer + "▌")
                        time.sleep(STREAM_DELAY)
                    elif event.type in {"response.output_text.done", "response.completed"}:
                        placeholder.markdown(answer)
            return answer
        except Exception:
            response = client.responses.create(**kwargs)
            answer = response.output_text
            placeholder.markdown(answer)
            return answer
    else:
        kwargs = {"model": model, "instructions": system, "input": prompt}
        with client.responses.stream(**kwargs) as stream:
            for event in stream:
                if event.type == "response.output_text.delta":
                    answer += event.delta
                    placeholder.markdown(answer + "▌")
                    time.sleep(STREAM_DELAY)
                elif event.type in {"response.output_text.done", "response.completed"}:
                    placeholder.markdown(answer)
                    
    user_content = prompt
    if images and model in _IMAGE_MODELS:
        content = [{"type": "text", "text": prompt}]
        for img in images:
            content.append(
                {"type": "text", "text": f"[uploaded/{img['name']} | image]:"}
            )
            content.append({"type": "image_url", "image_url": {"url": img["data_url"]}})
        user_content = content

    with client.chat.completions.stream(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
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
