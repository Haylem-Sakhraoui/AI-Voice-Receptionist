"""
Streamlit demo UI for the RAG assistant.
Calls the agent directly (no separate backend needed) so it works
both locally and when deployed on Streamlit Community Cloud.
"""
import sys
from pathlib import Path

# Make sure the project root (containing pipeline/) is importable,
# regardless of how the current working directory is set.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from pipeline.rag_chain import build_agent

st.set_page_config(page_title="Domain RAG Assistant", page_icon="🔧")
st.title("🔧 HVAC Knowledge Assistant (RAG Demo)")
st.caption("Powered by a fine-tuned PyTorch embedding model + LangChain retrieval pipeline")


@st.cache_resource
def get_agent():
    return build_agent()


agent_executor = get_agent()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if user_input := st.chat_input("Ask a question about HVAC service, pricing, or booking..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = agent_executor.invoke({"input": user_input})
                reply = result["output"]
            except Exception as e:
                reply = f"Sorry, something went wrong: {e}"
            st.write(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})