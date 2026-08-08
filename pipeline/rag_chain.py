"""
Builds the conversational RAG agent: retrieval over the fine-tuned
embedding index + memory + tool-calling, orchestrated with LangChain.
"""
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools.retriever import create_retriever_tool

from pipeline.tools import ALL_TOOLS

load_dotenv()

FINE_TUNED_PATH = Path(__file__).parent.parent / "models" / "fine_tuned_embedder"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"

SYSTEM_PROMPT = """You are a helpful assistant for an HVAC service company.
Answer customer questions using the retrieved knowledge base context.
If you don't know something, say so rather than guessing.
Use the available tools to check appointment availability or pricing when relevant.
Keep answers concise and friendly, like a knowledgeable phone receptionist."""


def build_agent() -> AgentExecutor:
    model_path = str(FINE_TUNED_PATH) if FINE_TUNED_PATH.exists() else "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_path)

    vectordb = Chroma(persist_directory=str(CHROMA_DIR), embedding_function=embeddings)
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})

    retriever_tool = create_retriever_tool(
        retriever,
        name="hvac_knowledge_base",
        description="Search the HVAC FAQ knowledge base for answers about troubleshooting, pricing, and policies.",
    )

    tools = ALL_TOOLS + [retriever_tool]

    llm = ChatOpenAI(model="openai/gpt-oss-20b:free",
    temperature=0.2,
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://openrouter.ai/api/v1",)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    agent = create_openai_tools_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True)


if __name__ == "__main__":
    # quick manual test in the terminal
    agent_executor = build_agent()
    print("RAG assistant ready. Type 'exit' to quit.\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        result = agent_executor.invoke({"input": user_input})
        print(f"Assistant: {result['output']}\n")
