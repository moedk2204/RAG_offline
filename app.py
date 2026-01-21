import gradio as gr
import os
import shutil
from src.config import INPUTS_DIR
from src.ingest import ingest_file, ingest_directory
from src.vector_store import update_vector_store, load_vector_store
from src.llm import get_ollama_llm
from src.rag import create_rag_chain, get_retriever

# --- Logic Functions ---

def process_query(message, history):
    """
    Callback for the ChatInterface.
    """
    try:
        vectorstore = load_vector_store()
        if not vectorstore:
            return "⚠️ No knowledge base found. Please upload a PDF in the 'Knowledge Base' tab first."
        
        llm = get_ollama_llm()
        retriever = get_retriever(vectorstore)
        qa_chain = create_rag_chain(llm, retriever)
        
        # Invoke chain
        response = qa_chain.invoke({"query": message})
        return response['result']
        
    except Exception as e:
        return f"❌ Error: {str(e)}"

def process_upload(files):
    """
    Callback for file upload.
    """
    if not files:
        return "⚠️ No files selected."
        
    status_msg = []
    try:
        all_docs = []
        for file in files:
            # Gradio passes file paths as named temporary files or direct paths
            temp_path = file.name
            filename = os.path.basename(temp_path)
            
            # Destination path
            save_path = os.path.join(INPUTS_DIR, filename)
            
            # Copy file to data/inputs
            shutil.copy2(temp_path, save_path)
            
            status_msg.append(f"📄 Saved to {filename} and processing...")
            
            # Simple check if regular file
            docs = ingest_file(save_path)
            all_docs.extend(docs)
            status_msg.append(f"   ✓ Extracted {len(docs)} chunks.")

        if all_docs:
            status_msg.append("💾 Updating Vector Store...")
            # Capture the status message returned by update_vector_store
            ingestion_status = update_vector_store(all_docs)
            status_msg.append(f"✅ {ingestion_status}")
            status_msg.append("✅ Ingestion Process Finished!")
        else:
            status_msg.append("⚠️ No content extracted.")
            
        return "\n".join(status_msg)
        
    except Exception as e:
        return f"❌ Error during ingestion: {str(e)}"

# --- UI Construction ---

# Custom CSS for a more modern look
custom_css = """
#component-0 {max-width: 1200px; margin: auto;}
.gradio-container {font-family: 'Inter', sans-serif;}
"""

# Modern Theme
theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="blue",
    neutral_hue="slate",
)

with gr.Blocks(title="RAG Assistant") as demo:
    
    with gr.Row():
        gr.Markdown(
            """
            # 🤖 AI Knowledge Assistant
            ### Chat with your documents using local RAG & Ollama
            """
        )
    
    with gr.Tabs():
        # --- Tab 1: Chat Interface ---
        with gr.TabItem("💬 Chat"):
            chat_interface = gr.ChatInterface(
                fn=process_query,
                chatbot=gr.Chatbot(height=600),
                textbox=gr.Textbox(
                    placeholder="Ask a question about your documents...", 
                    container=False, 
                    scale=7
                ),
            )
            
        # --- Tab 2: Ingestion ---
        with gr.TabItem("📚 Knowledge Base"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 📤 Upload Documents")
                    gr.Markdown("Upload **PDF files** here to add them to your knowledge base.")
                    
                    file_upload = gr.File(
                        file_count="multiple",
                        file_types=[".pdf"],
                        label="Select PDF Files"
                    )
                    upload_btn = gr.Button("🚀 Process Documents", variant="primary", size="lg")
                
                with gr.Column(scale=1):
                    gr.Markdown("### 📝 Status Log")
                    status_output = gr.Textbox(
                        label="Ingestion Status", 
                        lines=15, 
                        interactive=False
                    )
            
            upload_btn.click(
                fn=process_upload,
                inputs=[file_upload],
                outputs=[status_output]
            )

    with gr.Row():
        gr.Markdown(
            """
            <div style="text-align: center; color: #666; font-size: 0.9em; margin-top: 20px;">
                Powered by <b>LangChain</b>, <b>Ollama</b>, and <b>Gradio</b> 🚀
            </div>
            """
        )

if __name__ == "__main__":
    demo.launch(theme=theme, css=custom_css)
