import os

import streamlit as st
from mlflow.deployments import get_deploy_client

SERVING_ENDPOINT = os.getenv("SERVING_ENDPOINT", "databricks-gpt-5-5")

st.set_page_config(page_title="Databricks GPT-5.5 Chatbot", page_icon="🧱")
st.title("🧱 Databricks GPT-5.5 Chatbot")
st.caption(f"Streaming responses from `{SERVING_ENDPOINT}` via Unity AI Gateway.")


def stream_response(messages):
    client = get_deploy_client("databricks")
    for chunk in client.predict_stream(
        endpoint=SERVING_ENDPOINT,
        inputs={"messages": messages},
    ):
        choices = chunk.get("choices") or []
        if not choices:
            continue
        delta = choices[0].get("delta") or {}
        content = delta.get("content")
        if content:
            yield content


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = st.write_stream(stream_response(st.session_state.messages))

    st.session_state.messages.append({"role": "assistant", "content": response})
