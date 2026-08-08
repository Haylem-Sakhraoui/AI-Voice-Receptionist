# Domain-Specific RAG Assistant with Fine-Tuned Retrieval

A retrieval-augmented question-answering system that combines a **custom fine-tuned embedding model (PyTorch)** with an **orchestrated retrieval-generation pipeline (LangChain)** to answer questions accurately from a domain-specific document set — outperforming generic off-the-shelf embeddings on retrieval accuracy.

Unlike a typical "LLM wrapper" RAG demo, this project fine-tunes its own retrieval model on domain data and benchmarks it against a generic baseline, closing the gap between "used an API" and "built and evaluated a model."

---

## 🎯 Problem

Generic embedding models are trained on broad, general-purpose text and often underperform when retrieving information from specialized domains (technical manuals, contracts, FAQs) where terminology and phrasing don't match everyday language. This project demonstrates that a small, domain-fine-tuned embedding model can meaningfully improve retrieval accuracy over a generic baseline, directly improving the quality of the final AI-generated answer.

## ✨ Features

- **Custom-trained retrieval model** — a sentence-embedding model fine-tuned with contrastive learning on domain-specific query/passage pairs (PyTorch + `sentence-transformers`)
- **Full RAG pipeline** — document ingestion, chunking, embedding, vector storage, retrieval, and reranking, orchestrated with **LangChain**
- **Conversational memory** — multi-turn Q&A that keeps context across a conversation
- **Tool-calling** — the agent can call external tools (e.g., a mock "check availability" / "get pricing" function) mid-conversation
- **Evaluation suite** — a benchmark comparing the fine-tuned model against a generic baseline embedding model on retrieval recall@k and answer relevance
- **Deployed demo** — FastAPI backend + Streamlit frontend, containerized and runnable locally or deployed to a free-tier host

## 🏗️ Architecture

```
                ┌─────────────────────┐
                │   Raw domain docs    │
                └──────────┬───────────┘
                           │  chunking
                           ▼
                ┌─────────────────────┐
                │   Fine-tuned         │◄── trained with PyTorch on
                │   embedding model    │    domain query/passage pairs
                └──────────┬───────────┘
                           │  embeddings
                           ▼
                ┌─────────────────────┐
                │  Vector store        │  (Chroma / pgvector)
                └──────────┬───────────┘
                           │  top-k retrieval + rerank
                           ▼
                ┌─────────────────────┐
                │  LangChain pipeline  │  (retrieval chain, memory, tools)
                └──────────┬───────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │   LLM (generation)   │
                └──────────┬───────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │  FastAPI + Streamlit │
                └─────────────────────┘
```

## 🧪 Results

| Model | Recall@5 | MRR |
|---|---|---|
| Generic embedding baseline (e.g., `all-MiniLM-L6-v2`) | 0.71 | 0.58 |
| **Fine-tuned domain embedding (this project)** | **0.84** | **0.69** |

*Fine-tuning on ~[N] domain query/passage pairs improved retrieval recall@5 by ~13 points over the generic baseline, measured on a held-out test set of [N] domain-specific questions.*

*(Replace the numbers above with your actual measured results once you run the evaluation script — that's the most important part of the whole README.)*

## 🛠️ Tech Stack

- **PyTorch** + `sentence-transformers` — embedding model fine-tuning (contrastive loss)
- **LangChain** — retrieval pipeline orchestration, memory, tool-calling
- **Chroma** (or pgvector/Supabase) — vector storage
- **FastAPI** — backend API
- **Streamlit** — demo frontend
- **OpenAI / Anthropic API** — final answer generation (LLM)
- **Docker** — containerized deployment

## 📁 Repository Structure

```
domain-rag-assistant/
├── data/
│   ├── raw_documents/          # source domain documents
│   └── training_pairs.jsonl    # query-passage pairs for fine-tuning
├── training/
│   ├── fine_tune_embeddings.py # PyTorch contrastive fine-tuning script
│   └── evaluate_retrieval.py   # baseline vs fine-tuned comparison
├── pipeline/
│   ├── ingest.py               # chunking + embedding + vector store loading
│   ├── rag_chain.py            # LangChain retrieval + generation chain
│   └── tools.py                # custom tool-calling functions
├── app/
│   ├── main.py                 # FastAPI backend
│   └── streamlit_app.py        # demo UI
├── notebooks/
│   └── results_analysis.ipynb  # evaluation charts
├── Dockerfile
├── requirements.txt
└── README.md
```

## 🚀 Running Locally

```bash
git clone https://github.com/<your-username>/domain-rag-assistant.git
cd domain-rag-assistant
pip install -r requirements.txt

# 1. Fine-tune the embedding model
python training/fine_tune_embeddings.py

# 2. Ingest documents and build the vector index
python pipeline/ingest.py

# 3. Run the evaluation (fine-tuned vs baseline)
python training/evaluate_retrieval.py

# 4. Launch the demo
uvicorn app.main:app --reload &
streamlit run app/streamlit_app.py
```

## 📈 What This Project Demonstrates

- Applied ML: fine-tuning a neural embedding model with a custom training loop in PyTorch, not just calling a pretrained API
- LLM application engineering: building a production-style RAG pipeline with LangChain (chunking, retrieval, reranking, memory, tools)
- ML evaluation discipline: benchmarking a model change with a real held-out test set and reporting the delta
- End-to-end delivery: from raw data to a deployed, usable demo

---

*Built as a personal/portfolio project to explore fine-tuned retrieval for domain-specific RAG systems.*
