# 🤖 RAG Chatbot (Ollama API + LangChain + Gradio)

### 🚀 [Live Demo on Hugging Face Spaces](https://huggingface.co/spaces/MD2204/RAG_chatPDF)

This repository contains a **Retrieval Augmented Generation (RAG)** system built in **Python** that allows you to chat with your PDF documents.  
It uses the **Ollama API** (powered by `gpt-oss:120b`), **FAISS** for the vector database, and provides a modern **Gradio** web interface.

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
├─ app.py                # Modern Web UI (Gradio)
├─ main.py               # CLI Entry point
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
- [Ollama](https://ollama.com/) (running locally or accessible via API)
- [Ollama Model] `gpt-oss:120b` (or any other model you configure)

Install dependencies:
```bash
pip install -r requirements.txt
```

> [!NOTE]
> Ensure you have pulled the model in Ollama before running:
> `ollama pull gpt-oss:120b`

---

## 🚀 Quick Start

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
> **Ollama Connection**: If Ollama is running on your host machine, update your `.env` to:  
> `OLLAMA_BASE_URL=http://host.docker.internal:11434`

### 4️⃣ Option D: Docker Hub (Ready-to-use Image)

If you want to pull a pre-built image (without source code):


  **Pull and Run (from Any PC)**:
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
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace).
- **Vector Store**: FAISS (Facebook AI Similarity Search).
- **Orchestration**: LangChain.

### ✨ Key Features

- **Persistence**: The vector database is saved to disk (`data/vector_db`). You don't need to re-ingest files after restarting.
- **Smart Deduplication**: The system checks file metadata. If a PDF is already in the DB, it skips re-processing it to avoid duplicate chunks.
- **Auto-Saving**: Uploaded files in Gradio are automatically copied to `data/inputs` for safekeeping.

---

## 📜 License
MIT — free to use for learning and building your own local AI assistants.
