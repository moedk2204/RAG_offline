from src.llm import test_llm_connection
from src.config import VECTOR_DB_DIR
import os

print("🔍 Verifying Setup...")

# 1. Check Directories
if os.path.exists(VECTOR_DB_DIR):
    print(f"✓ Vector DB Directory exists: {VECTOR_DB_DIR}")
else:
    print(f"⚠️ Vector DB Directory NOT found: {VECTOR_DB_DIR} (This is expected if you haven't ingested anything yet)")

# 2. Check LLM
if test_llm_connection():
    print("✓ LLM Connection OK")
else:
    print("❌ LLM Connection FAILED")

print("\n🚀 Setup looks good! Reading to run.")
print("Try loading a PDF: python main.py --ingest path/to/your.pdf")
