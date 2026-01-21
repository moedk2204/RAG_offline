import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
INPUTS_DIR = DATA_DIR / "inputs"
VECTOR_DB_DIR = DATA_DIR / "vector_db"

# Ensure directories exist
INPUTS_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)

# Ollama Settings
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gpt-oss:20b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0.1"))

# Model Settings
EMBEDDING_MODEL_NAME = "nomic-embed-text"
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu") 

# text splitter settings
CHUNK_SIZE = 700
CHUNK_OVERLAP = 50

# retrieval settings
RETRIEVER_K = 3
