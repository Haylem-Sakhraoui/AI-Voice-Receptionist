"""
FastAPI backend exposing the RAG agent as a /chat endpoint.
"""
from fastapi import FastAPI
from pydantic import BaseModel

from pipeline.rag_chain import build_agent

app = FastAPI(title="Domain RAG Assistant")

# Build once at startup - reused across requests
agent_executor = build_agent()


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    reply: str


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    result = agent_executor.invoke({"input": request.message})
    return ChatResponse(reply=result["output"])


@app.get("/health")
def health():
    return {"status": "ok"}
