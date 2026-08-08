"""
Simple Streamlit demo UI for the RAG assistant.
Calls the local FastAPI backend at /chat.
"""
import requests
import streamlit as st

API_URL = "http://localhost:8000/chat"

st.set_page_config(page_title="Domain RAG Assistant", page_icon="🔧")
st.title("🔧 HVAC Knowledge Assistant (RAG Demo)")
st.caption("Powered by a fine-tuned PyTorch embedding model + LangChain retrieval pipeline")

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
                response = requests.post(API_URL, json={"message": user_input}, timeout=30)
                reply = response.json()["reply"]
            except Exception as e:
                reply = f"Error reaching backend: {e}"
            st.write(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})
