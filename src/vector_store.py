import os
import shutil
from typing import List, Optional
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from src.config import (
    EMBEDDING_MODEL_NAME, 
    EMBEDDING_DEVICE, 
    VECTOR_DB_DIR
)

def get_embeddings():
    """Initialize HuggingFace embeddings."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': EMBEDDING_DEVICE},
        encode_kwargs={'normalize_embeddings': True}
    )

def load_vector_store() -> Optional[FAISS]:
    """Load existing FAISS index from disk."""
    if not os.path.exists(os.path.join(VECTOR_DB_DIR, "index.faiss")):
        return None
        
    embeddings = get_embeddings()
    # allow_dangerous_deserialization is needed for loading local files
    return FAISS.load_local(
        folder_path=str(VECTOR_DB_DIR), 
        embeddings=embeddings,
        allow_dangerous_deserialization=True 
    )

def create_vector_store(documents: List[Document]) -> FAISS:
    """Create a new vector store from documents."""
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore

def save_vector_store(vectorstore: FAISS):
    """Save vector store to disk."""
    vectorstore.save_local(folder_path=str(VECTOR_DB_DIR))

def get_existing_sources(vectorstore: FAISS) -> set:
    """Extract unique source paths from the existing vector store."""
    if not vectorstore:
        return set()
    
    unique_sources = set()
    # Access the docstore directly to check metadata
    # This assumes the docstore is in memory, which is true for default FAISS
    for doc_id, doc in vectorstore.docstore._dict.items():
        if 'source' in doc.metadata:
            # Normalize path (resolve symlinks, standardize slashes)
            normalized_source = os.path.normpath(os.path.abspath(doc.metadata['source']))
            unique_sources.add(normalized_source)
    
    return unique_sources

def update_vector_store(documents: List[Document]) -> FAISS:
    """
    Add new documents to the vector store.
    Skips documents that are already present based on their source path.
    """
    vectorstore = load_vector_store()
    
    if vectorstore:
        existing_sources = get_existing_sources(vectorstore)
        
        # Filter documents
        new_documents = []
        skipped_count = 0
        
        # We process by file source, not by chunk, to ensure consistency
        # We assume all chunks from the same file share the same source
        
        # Helper to track which files we've already decided to add in this batch
        batch_new_sources = set()

        for doc in documents:
            source = doc.metadata.get('source')
            if source:
                # Normalize path for comparison
                normalized_source = os.path.normpath(os.path.abspath(source))
                if normalized_source in existing_sources:
                    skipped_count += 1
                    continue
            
            new_documents.append(doc)
            if source:
                batch_new_sources.add(source)
            
        if not new_documents:
            msg = f"ℹ️  No new documents to add. Skipped {skipped_count} chunks from existing files."
            print(msg)
            return msg
            
        msg = f"ℹ️  Adding {len(new_documents)} new chunks. Skipped {skipped_count} existing chunks."
        print(msg)
        vectorstore.add_documents(new_documents)
        
    else:
        msg = f"🆕 Creating new vector store with {len(documents)} chunks."
        print(msg)
        vectorstore = create_vector_store(documents)
        
    save_vector_store(vectorstore)
    return msg
