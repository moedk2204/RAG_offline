import argparse
import sys
import os
from src.config import INPUTS_DIR
from src.ingest import ingest_file, ingest_directory
from src.vector_store import update_vector_store, load_vector_store
from src.llm import get_ollama_llm, test_llm_connection
from src.rag import create_rag_chain, get_retriever

def ingest_path(path: str):
    """Ingest a PDF file or directory of PDFs."""
    print(f"📄 Processing: {path}")
    try:
        if os.path.isdir(path):
            documents = ingest_directory(path)
            print(f"✓ Loaded {len(documents)} chunks from directory")
        else:
            documents = ingest_file(path)
            print(f"✓ Split into {len(documents)} chunks")
        
        print("💾 Updating Vector Store...")
        update_vector_store(documents)
        print("✓ Ingestion Complete!")
    except Exception as e:
        print(f"❌ Error: {e}")

def run_chat_loop():
    """Interactive chat loop."""
    print("🤖 Initializing RAG System...")
    
    if not test_llm_connection():
        print("❌ Cannot connect to Ollama. Exiting.")
        return

    vectorstore = load_vector_store()
    if not vectorstore:
        print("⚠️ No vector store found. Please ingest a PDF first.")
        print("Usage: python main.py --ingest path/to/file.pdf")
        return

    llm = get_ollama_llm()
    retriever = get_retriever(vectorstore)
    qa_chain = create_rag_chain(llm, retriever)
    
    print("\n💬 Chat with your PDF (Type 'exit' to quit)")
    print("-" * 50)
    
    while True:
        try:
            query = input("\nYou: ")
            if query.lower() in ['exit', 'quit', 'q']:
                break
            
            print("AI: Thinking...", end="\r")
            result = qa_chain.invoke({"query": query})
            
            print(f"AI: {result['result']}")
            
            # Optional: Show sources
            # for doc in result['source_documents']:
            #     print(f"   (Source: {doc.metadata.get('source', 'unknown')})")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def main():
    parser = argparse.ArgumentParser(description="RAG PDF Assistant")
    parser.add_argument("--ingest", type=str, help="Path to PDF file to ingest")
    parser.add_argument("--query", type=str, help="Single query to run")
    
    args = parser.parse_args()
    
    if args.ingest:
        ingest_path(args.ingest)
    elif args.query:
        # One-off query
        vectorstore = load_vector_store()
        if not vectorstore:
            print("❌ No vector store found.")
            return
        llm = get_ollama_llm()
        retriever = get_retriever(vectorstore)
        qa_chain = create_rag_chain(llm, retriever)
        result = qa_chain.invoke({"query": args.query})
        print(result['result'])
    else:
        run_chat_loop()

if __name__ == "__main__":
    main()
