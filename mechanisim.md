# How This Project Works — A Beginner's Guide

A walkthrough of the RAG PDF Chat codebase, written to be explained to someone who has never built an AI app before.

---

## The one-sentence version

This app lets you **ask questions about your own PDFs**. A normal chatbot doesn't know what's in your documents. This one first *finds* the relevant paragraphs in your PDFs, then hands them to the AI model along with your question. That pattern is called **RAG** — Retrieval-Augmented Generation.

## The analogy to use with beginners

Imagine an open-book exam.

- The AI model is the **student**. Smart, but has never seen your textbook.
- Your PDFs are the **textbook**.
- RAG is the **assistant who flips to the right 3 pages** and slides them under the student's nose before they answer.

Everything in this codebase is either "prepare the textbook" or "flip to the right pages."

---

## The two phases

### Phase 1 — Ingestion (done once per PDF)

```
PDF → text → small chunks → numbers (vectors) → saved to disk
```

| Step | File | What happens |
|---|---|---|
| Read the PDF | `src/ingest.py:14` | `PyPDFLoader` pulls raw text out of the PDF |
| Cut it up | `src/ingest.py:27` | Splits text into ~700-character chunks, with 50 chars overlapping so sentences aren't sliced in half |
| Turn text into numbers | `src/vector_store.py:15` | An *embedding model* converts each chunk into a list of numbers that represent its **meaning** |
| Store it | `src/vector_store.py:53` | FAISS saves those numbers to `data/vector_db/` so you never redo this |

**Why chunks?** You can't shove a 300-page PDF into an AI prompt — there's a size limit. Small chunks let you retrieve only the relevant bits.

**Why numbers?** This is the key idea for beginners. An embedding turns text into coordinates in "meaning space." "dog" and "puppy" end up close together; "dog" and "tax return" end up far apart. That means you can search by *meaning*, not by exact keyword matching like Ctrl+F.

There's also a nice touch in `src/vector_store.py:73` — before adding a PDF it checks whether that file's path is already in the database, so re-uploading the same PDF doesn't duplicate everything.

### Phase 2 — Answering (every question)

```
your question → numbers → find 3 closest chunks → stuff into prompt → LLM → answer
```

| Step | File | What happens |
|---|---|---|
| Retrieve | `src/rag.py:9` | Your question is embedded too, then FAISS returns the `k=3` nearest chunks |
| Build the prompt | `src/rag.py:17` | Chunks get pasted into a template: *"Use the following context to answer... if you don't know, say so"* |
| Generate | `src/llm.py:11` | The filled-in prompt goes to Ollama, which returns the answer |

That "if you don't know, just say you don't know" line is the anti-hallucination guardrail. Worth pointing out to beginners — it's a single sentence of English doing real engineering work.

---

## The file map

```
app.py              ← Gradio web UI (2 tabs: Chat + upload)
main.py             ← same thing, but terminal-based
src/config.py       ← all the knobs: model name, chunk size, k, paths
src/ingest.py       ← PDF → chunks
src/vector_store.py ← chunks → vectors → FAISS on disk
src/llm.py          ← connection to Ollama
src/rag.py          ← glues retriever + prompt + LLM together
data/inputs/        ← your PDFs
data/vector_db/     ← the searchable index
```

The teaching point here: **each file does one job**, and `config.py` holds every tunable setting so nothing is hardcoded in five places. Change `CHUNK_SIZE` or `RETRIEVER_K` in one spot and the whole system changes.

## The libraries, in plain terms

- **LangChain** — the plumbing. Provides ready-made PDF loaders, splitters, and the `RetrievalQA` chain that wires retrieval → prompt → LLM.
- **FAISS** — the vector database. A file-based search engine for "find me the closest vectors."
- **Ollama** — runs the actual language model. It's a separate program on your machine, listening on port `11434`. Your Python app just makes HTTP calls to it.
- **Gradio** — turns Python functions into a web UI. `app.py` is basically two functions (`process_query`, `process_upload`) with buttons attached.

The important consequence: **with local models, nothing leaves your computer.** No API keys, no cloud, no per-token billing. Good selling point when explaining it.

One honest caveat, since it's easy to lose track of: that's only true while every model is local. Ollama also offers `-cloud` model tags that run on remote hardware — pick one of those as your chat model and your document chunks *are* sent off your machine, even though the embeddings and the FAISS index stay local. Both setups are fine; just know which one you're demoing before you claim it's fully offline.

---

## How to launch it

**Step 0 — Install Ollama.** The Python app is useless without it, and it does *not* come with the project. Install it system-wide first:

```powershell
winget install Ollama.Ollama
```

(Or download the Windows installer from https://ollama.com/download/windows.) It installs as a background service listening on port `11434`, so you normally don't need to start it by hand.

> **Docker doesn't let you skip this.** `docker-compose.yml` builds only the app container — there is no Ollama service in it, and `host.docker.internal` points back at *your machine*. Ollama must be installed natively either way.

**Step 0b — Pull the models.**

```powershell
ollama serve                     # only if the service isn't already running
ollama pull gemma4:31b-cloud     # the chat model
ollama pull nomic-embed-text     # the embedding model — required, not optional
```

`nomic-embed-text` (~275MB) is not negotiable: `src/config.py:24` hardcodes it, so ingestion fails without it.

The chat model is your choice, but two things must line up:

- **The tag must match `.env`.** `OLLAMA_MODEL` in `.env` is what the code actually loads (`src/config.py:19`). Pulling one model and leaving a different name in `.env` is the most common way to get a confusing "model not found" error.
- **Watch the size.** Local models download in full — a 20B model is roughly 13GB and wants ~16GB of RAM. If you just want to confirm the pipeline works, `llama3.2:3b` (~2GB) is enough, then switch later. Run `ollama list` to see what you actually have.

> **A note on `-cloud` model tags:** those run on Ollama's servers rather than your machine. They skip the huge download and the RAM requirement, but they need `ollama signin` first, and your document text is sent to a remote service. See the privacy note below.

**Steps 1–4 — the app:**

```powershell
cd "C:\Users\MoEdBouk\Desktop\AI ENGENEER PATH\RAG_PDF_CHAT"

python -m venv venv               # 1. create the virtual environment
.\venv\Scripts\Activate.ps1       # 2. activate it (prompt now shows "(venv)")
pip install -r requirements.txt   # 3. install dependencies into it
python verify_setup.py            # 4. check Ollama answers before going further
python app.py                     # 5. run — opens http://127.0.0.1:7860
```

Step 4 is the one worth doing every time something breaks. `verify_setup.py` sends a single test message to Ollama and prints the reply — if that fails, the problem is your Ollama setup, not the RAG code, and there's no point debugging the UI.

Then: **Knowledge Base** tab → upload a PDF → *Process Documents* → switch to **Chat** and ask away.

CLI alternative:

```powershell
python main.py --ingest "data/inputs/MyDoc.pdf"   # load a PDF
python main.py                                    # interactive chat
python main.py --query "Summarize the report"     # one-off question
```

### Fix these before your first run

The first two will stop the app from starting at all — do them between step 3 and step 4.

1. **`.env` has `OLLAMA_BASE_URL=http://host.docker.internal:11434`.** That address only resolves *inside a Docker container*. Running `python app.py` directly on Windows, it can't connect — change it to `http://localhost:11434`. (Change it back if you later switch to `docker-compose`.)
2. **`src/rag.py:1` imports `langchain_classic`, which isn't in `requirements.txt`.** You'll get a `ModuleNotFoundError` on startup. Either `pip install langchain-classic` or change the import to `from langchain.chains import RetrievalQA`.
3. **`gemma4:31b-cloud` is a cloud tag, so `ollama signin` is required.** Without being signed in, the model resolves but every request fails with an auth error. Run `ollama list` to confirm the exact tag string — a single character off (`31bcloud` vs `31b-cloud`) produces a "model not found" that looks identical to not having pulled it at all.
4. **The README says embeddings are `all-MiniLM-L6-v2` (HuggingFace), but `config.py:24` sets `nomic-embed-text`**, which routes to Ollama instead. The code wins — so you do need that `ollama pull`. The HuggingFace path in `src/vector_store.py:27` is real but unreachable unless you edit the config, which is why `requirements.txt` still drags in `torch` and `sentence-transformers`.

---

## Why a venv? (the part beginners always ask)

**A virtual environment is a private box of libraries that belongs to this project only.**

Without one, `pip install` dumps everything into one system-wide Python. That causes real problems:

- **Version collisions.** This project needs a specific LangChain version. Another project needs a different one. One global Python can only hold one version at a time — installing for project B silently breaks project A.
- **No reproducibility.** `requirements.txt` is a promise: "install these and it works." That promise is only testable in a clean box. In a polluted global Python, it works on your machine and nowhere else.
- **It's not disposable.** Broken global install = painful surgery. Broken venv = delete the folder, recreate, 30 seconds.
- **It keeps things honest.** This project pulls in `torch`, which is a ~200MB download. You don't want that permanently welded into your system Python because you tried a tutorial once.

The analogy: a venv is a **separate toolbox per job**, instead of throwing every tool you've ever owned into one drawer and hoping the right wrench is on top.

Mechanically, `python -m venv venv` creates a `venv/` folder with its own copy of Python and its own empty `site-packages`. Activating it just puts that folder first on your `PATH`, so `python` and `pip` now mean *the ones in the box*. `deactivate` puts things back. And `venv/` is in `.gitignore` — you commit the *recipe* (`requirements.txt`), never the ingredients.

> Docker, by the way, is the same idea taken further: a venv isolates Python packages; Docker isolates the whole operating system. That's why `docker-compose up --build` runs this app without you installing Python at all — but it still doesn't bundle Ollama, so that install stays on your host machine either way.
