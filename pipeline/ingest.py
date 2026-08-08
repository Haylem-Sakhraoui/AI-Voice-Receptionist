"""
Ingest domain documents: split into chunks, embed with the fine-tuned
model, and store in a persistent Chroma vector database.
"""
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

DOCS_DIR = Path(__file__).parent.parent / "data" / "raw_documents"
FINE_TUNED_PATH = Path(__file__).parent.parent / "models" / "fine_tuned_embedder"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"


def get_embedding_model():
    model_path = str(FINE_TUNED_PATH) if FINE_TUNED_PATH.exists() else "sentence-transformers/all-MiniLM-L6-v2"
    print(f"Using embedding model: {model_path}")
    return HuggingFaceEmbeddings(model_name=model_path)


def load_and_chunk_documents() -> list[str]:
    all_text = ""
    for file in DOCS_DIR.glob("*.txt"):
        all_text += file.read_text() + "\n"

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50,
        separators=["\nQ:", "\n\n", "\n", " "],
    )
    return splitter.split_text(all_text)


def main():
    chunks = load_and_chunk_documents()
    print(f"Split documents into {len(chunks)} chunks")

    embeddings = get_embedding_model()

    vectordb = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
    )
    vectordb.persist()
    print(f"Vector store built and saved to {CHROMA_DIR}")


if __name__ == "__main__":
    main()
