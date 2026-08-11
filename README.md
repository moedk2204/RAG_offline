# 🤖 RAG Chatbot (Ollama API + LangChain + Gradio)

### 🚀 [Live Demo on Hugging Face Spaces](https://huggingface.co/spaces/MD2204/RAG_chatPDF)

This repository contains a **Retrieval Augmented Generation (RAG)** system built in **Python** that allows you to chat with your PDF documents.  
It uses the **Ollama API** (powered by `gemma4:31b-cloud`), **FAISS** for the vector database, and provides a modern **Gradio** web interface.

> [!NOTE]
> **Where your data goes:** PDF parsing, embeddings, and the FAISS index all stay on your machine. The chat model is whatever `OLLAMA_MODEL` points at — the default `gemma4:31b-cloud` is a *cloud* tag, so the retrieved text chunks are sent to Ollama's servers. Swap in a local tag (e.g. `llama3.2:3b`) for a fully offline setup.

It is designed as a **learning project** for understanding:

- RAG pipelines (Ingestion -> Embedding -> Retrieval -> Generation)
- Vector Databases (FAISS)
- LLM integration (Ollama API)
- Building interactive AI UIs

---



## 📂 Project Structure

```
.
├─ README.md
├─ requirements.txt
├─ Dockerfile            # Docker recipe
├─ docker-compose.yml    # Docker orchestration
├─ .env.example
├─ .dockerignore
├─ app.py                # Modern Web UI (Gradio)
├─ main.py               # CLI Entry point
├─ verify_setup.py       # Connection / setup check
├─ mechanisim.md         # Beginner-friendly walkthrough of how it works
├─ src/
│ ├─ __init__.py
│ ├─ config.py           # Configuration (Paths, Models)
│ ├─ ingest.py           # PDF Loader & Text Splitter
│ ├─ vector_store.py     # FAISS Vector DB & Embeddings
│ ├─ llm.py              # Ollama Connector
│ └─ rag.py              # RAG Chain Construction
└─ data/
  ├─ inputs/             # Store your PDFs here
  └─ vector_db/          # Persistent FAISS index
```

---



## ⚙️ Requirements

- Python 3.10+
- [Ollama](https://ollama.com/) installed and running (it is **not** bundled with this project, and the Docker option does not include it either)
- Chat model: `gemma4:31b-cloud` (or any other model you configure)
- Embedding model: `nomic-embed-text` — required, set in `src/config.py`

Install dependencies:

```bash
pip install -r requirements.txt
pip install langchain-classic          # imported by src/rag.py, missing from requirements.txt
```

Pull the models:

```bash
ollama pull gemma4:31b-cloud
ollama pull nomic-embed-text
```

> [!IMPORTANT]
> `gemma4:31b-cloud` is a **cloud** model tag — it runs on Ollama's servers, not your machine. Run `ollama signin` first, or every request fails with an authentication error. Run `ollama list` to confirm the exact tag you have.
>
> To run fully offline instead, set `OLLAMA_MODEL` in `.env` to a local tag such as `llama3.2:3b` and pull that.

---



## 🚀 Quick Start



First, copy `.env.example` to `.env` and set `OLLAMA_BASE_URL` for how you're running:

| How you run it | `OLLAMA_BASE_URL` |
|---|---|
| Option A or B (Python on your machine) | `http://localhost:11434` |
| Option C or D (inside Docker) | `http://host.docker.internal:11434` |

> [!WARNING]
> This is the most common setup failure. `host.docker.internal` does not resolve outside a container, so leaving it set while running `python app.py` produces a connection error that looks like Ollama is down.

Verify the connection before launching the UI:

```bash
python verify_setup.py
```

### 1️⃣ Option A: Web Interface (Recommended)

Launch the modern Chat UI:

```bash
python app.py
```

This will open a local URL (e.g., `http://127.0.0.1:7860`).

- **Chat Tab**: Ask questions about your docs.
- **Knowledge Base Tab**: Upload PDFs. They are automatically saved, ingested, and deduplicated.



### 2️⃣ Option B: Command Line (CLI)

Chat interactively in the terminal:

```bash
python main.py
```

Ingest a single PDF or Directory:

```bash
python main.py --ingest "data/inputs/MyDoc.pdf"
# OR
python main.py --ingest "data/inputs"
```

Perform a single query:

```bash
python main.py --query "What is the summary of the report?"
```



### 3️⃣ Option C: Docker (Containerized)

If you have Docker installed, you can run the entire interface without installing Python locally:

```bash
docker-compose up --build
```

Access the UI at `http://localhost:7860`.

> [!IMPORTANT]
> **Ollama is still required on the host.** `docker-compose.yml` builds only the app container — it contains no Ollama service. Install Ollama natively, then point the container back at your machine in `.env`:  
> `OLLAMA_BASE_URL=http://host.docker.internal:11434`



### 4️⃣ Option D: Docker Hub (Ready-to-use Image)

If you want to pull a pre-built image (without source code), save this as `docker-compose.yml`:

```yaml
version: '3.8'
services:
  rag-app:
    image: mohamad220/chat-rag-ai:latest
    ports:
      - "7860:7860"
    volumes:
      - ./data:/app/data
      - hf_cache:/app/data/hf_cache
    environment:
      - OLLAMA_MODEL=gemma4:31b-cloud
      - OLLAMA_BASE_URL=http://host.docker.internal:11434
      - EMBEDDING_DEVICE=cpu
    extra_hosts:
      - "host.docker.internal:host-gateway"

volumes:
  hf_cache:
```

Then run:

```bash
docker compose up
```

---



## 🛠️ Implementation Notes



### 🧩 Components

- **LLM**: Ollama (configured in `.env` or `src/config.py`).
- **Embeddings**: `nomic-embed-text` served by Ollama. A HuggingFace path (`all-MiniLM-L6-v2`) also exists in `src/vector_store.py` and takes over if you change `EMBEDDING_MODEL_NAME` — that's why `torch` and `sentence-transformers` are still in `requirements.txt`.
- **Vector Store**: FAISS (Facebook AI Similarity Search).
- **Orchestration**: LangChain.



### ✨ Key Features

- **Persistence**: The vector database is saved to disk (`data/vector_db`). You don't need to re-ingest files after restarting.
- **Smart Deduplication**: The system checks file metadata. If a PDF is already in the DB, it skips re-processing it to avoid duplicate chunks.
- **Auto-Saving**: Uploaded files in Gradio are automatically copied to `data/inputs` for safekeeping.

---



## 📜 License

MIT — free to use for learning and building your own AI assistants.