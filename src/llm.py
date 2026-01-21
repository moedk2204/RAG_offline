from langchain_community.chat_models import ChatOllama
from src.config import (
    OLLAMA_MODEL,
    OLLAMA_BASE_URL,
    OLLAMA_TEMPERATURE
)

def get_ollama_llm() -> ChatOllama:
    """Initialize ChatOllama instance (offline)."""

    return ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,  # http://localhost:11434
        temperature=OLLAMA_TEMPERATURE,
    )

def test_llm_connection() -> bool:
    """Test connection to local Ollama."""
    try:
        print(f"Testing Ollama at {OLLAMA_BASE_URL} with model {OLLAMA_MODEL}...")
        llm = get_ollama_llm()
        response = llm.invoke("Hello, are you running locally?")
        print(f"✓ Connection successful.\nResponse:\n{response.content}")
        return True
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False
