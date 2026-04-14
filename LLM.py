import streamlit as st
import openai
import anthropic
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter 
import tempfile
import os

# --- 1. CONFIGURATION & SECRETS ---
API_KEYS = {
    "Groq-Llama-4": st.secrets.get("GROQ_KEY", "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"),
    "Claude-3-Haiku": st.secrets.get("ANTHROPIC_KEY", "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"),
    "gemini-3-flash-preview": st.secrets.get("GOOGLE_KEY", "AIzaSyBmPU22hpyfCcy8u0wJiMbj6WGwQii8mWU"),
}

st.set_page_config(page_title="Universal AI Hub 2026", layout="wide")

# --- 2. SESSION STATE INITIALIZATION ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "artifacts" not in st.session_state:
    st.session_state.artifacts = {}
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "uploaded_pdf" not in st.session_state:
    st.session_state.uploaded_pdf = None

# --- 3. HELPER FUNCTIONS (RAG & UTILS) ---
@st.cache_resource
def process_pdf_rag(uploaded_file, google_api_key):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name
    
    loader = PyPDFLoader(tmp_path)
    pages = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(pages)
    
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", 
        google_api_key=google_api_key
    )
    vectorstore = FAISS.from_documents(docs, embeddings)
    os.remove(tmp_path)
    return vectorstore

def get_rag_context(vectorstore, prompt):
    if vectorstore is None:
        return ""
    search_results = vectorstore.similarity_search(prompt, k=3)
    return "\n".join([d.page_content for d in search_results])

# --- 4. SIDEBAR NAVIGATION ---
with st.sidebar:
    st.title("🚀 Multi-AIChatbot Pro")
    mode = st.radio("Switch Mode", ["💬 Multi-Chat + RAG", "🎨 Image Gen", "🎬 Video Gen"])
    
    st.divider()
    st.subheader("📁 Attachments")
    uploaded_file = st.file_uploader("Upload PDF for RAG", type=['pdf'], key="pdf_uploader")
    
    if uploaded_file is not None and uploaded_file != st.session_state.uploaded_pdf:
        if "gemini-3-flash-preview" in API_KEYS:
            with st.status("Processing PDF with RAG..."):
                st.session_state.vectorstore = process_pdf_rag(uploaded_file, API_KEYS["gemini-3-flash-preview"])
                st.session_state.uploaded_pdf = uploaded_file
                st.success("PDF processed! RAG ready.")
        else:
            st.warning("Upload Google API key in secrets for RAG.")
    
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.session_state.vectorstore = None
        st.session_state.uploaded_pdf = None
        st.rerun()

    st.divider()
    st.subheader("📦 Recent Artifacts")
    for name in st.session_state.artifacts.keys():
        if st.button(f"📄 {name}", key=f"art_{name}"):
            st.info("Artifact content is stored in memory.")

# --- 5. MAIN INTERFACE ---
if mode == "💬 Multi-Chat + RAG":
    selected_model = st.selectbox("Choose Brain", list(API_KEYS.keys()))
    
    # Display Chat History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat Input & Response Generation
    if prompt := st.chat_input("Say something..."):    
        # Add user message immediately
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate assistant response
        with st.chat_message("assistant"):
            status = st.status("Thinking...", expanded=True)
            
            try:
                # RAG Context
                rag_context = get_rag_context(st.session_state.vectorstore, prompt)
                if rag_context:
                    full_prompt = f"Context from documents:\n{rag_context}\n\nUser Question: {prompt}"
                else:
                    full_prompt = prompt
                
                # Build full conversation history for context
                messages = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
                
                # API Routing
                if "gemini" in selected_model.lower():
                    genai.configure(api_key=API_KEYS["gemini-3-flash-preview"])
                    model = genai.GenerativeModel('gemini-3-flash-preview')
                    response = model.generate_content(full_prompt)
                    res_text = response.text

                elif "Groq" in selected_model:
                    client = Groq(api_key=API_KEYS["Groq-Llama-4"])
                    response = client.chat.completions.create(
                        model="meta-llama/llama-4-scout-17b-16e-instruct",  
                        messages=[{"role": "user", "content": prompt}]
                    )
                    res_text = response.choices[0].message.content

                elif "claude" in selected_model.lower():
                    client = anthropic.Anthropic(api_key=API_KEYS["Claude-3-Haiku"])
                    response = client.messages.create(
                        model="claude-3-haiku-4-5-20251001",  # Updated 2026 model ID
                        max_tokens=1024,
                        messages=[{"role": "user", "content": full_prompt}]
                    )
                    res_text = response.content[0].text

                status.update(label="Response Complete!", state="complete", expanded=False)
                st.markdown(res_text)
                
                # Save to history
                st.session_state.messages.append({"role": "assistant", "content": res_text})
                
                # Auto-Artifact for code
                if "```" in res_text:
                    art_name = f"Code_{len(st.session_state.artifacts)+1}"
                    st.session_state.artifacts[art_name] = res_text
                    st.info(f"💾 Code saved as artifact: {art_name}")

            except Exception as e:
                status.update(label="Error!", state="error")
                st.error(f"API Error: {str(e)}")

elif mode == "🎨 Image Gen":
    st.header("Image Generation (Coming Soon)")
    img_prompt = st.text_area("Describe the image...")
    if st.button("Generate Image"):
        st.info("Image gen logic to be implemented with Gemini/DALL-E.")

elif mode == "🎬 Video Gen":
    st.header("Video Generation (Coming Soon)")
    vid_prompt = st.text_input("Describe the video scene...")
    if st.button("Generate Video"):
        st.info("Video gen logic to be implemented.")

# Instructions
with st.expander("📋 Setup Instructions"):
    st.markdown("""
     
        
