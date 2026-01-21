from langchain_classic.chains import RetrievalQA  
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.language_models import BaseChatModel
from src.config import RETRIEVER_K

def get_retriever(vectorstore) -> BaseRetriever:
    """Get retriever from vector store."""
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": RETRIEVER_K}
    )

def create_rag_chain(llm: BaseChatModel, retriever: BaseRetriever):
    """Create RAG chain."""
    
    prompt_template = """Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    Context:
    {context}

    Question: {question}
    Answer:"""
    
    PROMPT = PromptTemplate(
        template=prompt_template, 
        input_variables=["context", "question"]
    )
     
    # Using RetrievalQA as in the notebook (or we could use modern LCEL chains)
    # Keeping it simple for now as per the notebook's approach
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    
    return qa_chain
