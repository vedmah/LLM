import streamlit as st
import openai
import anthropic
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# FIXED IMPORT FOR 2026
from langchain_text_splitters import RecursiveCharacterTextSplitter 

import tempfile
import os
 
# --- 1. CONFIGURATION & SECRETS ---
# Note: Ensure these keys are set in Streamlit Cloud "Secrets" or local .streamlit/secrets.toml
API_KEYS = {
    "GPT-4o-mini": st.secrets.get("OPENAI_KEY", "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" ),
    "Claude-3-Haiku": st.secrets.get("ANTHROPIC_KEY", " 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" ),
    "gemini-3-flash-preview": st.secrets.get("GOOGLE_KEY", "AIzaSyBmPU22hpyfCcy8u0wJiMbj6WGwQii8mWU"),
     
}

st.set_page_config(page_title="Universal AI Hub 2026", layout="wide")

# --- 2. SESSION STATE INITIALIZATION ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "artifacts" not in st.session_state:
    st.session_state.artifacts = {}

# --- 3. HELPER FUNCTIONS (RAG & UTILS) ---
def process_pdf_rag(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name
    
    loader = PyPDFLoader(tmp_path)
    pages = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(pages)
    
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", 
        google_api_key=API_KEYS["Gemini-1.5-Flash"]
    )
    vectorstore = FAISS.from_documents(docs, embeddings)
    os.remove(tmp_path)
    return vectorstore

# --- 4. SIDEBAR NAVIGATION ---
with st.sidebar:
    st.title("🚀 Multi-AIChatbot Pro")
    mode = st.radio("Switch Mode", ["💬 Multi-Chat + RAG", "🎨 Image Gen", "🎬 Video Gen"])
    
    st.divider()
    st.subheader("📁 Attachments")
    uploaded_file = st.file_uploader("Upload PDF, Image, or Video", type=['pdf', 'png', 'jpg', 'mp4'])
    
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.subheader("📦 Recent Artifacts")
    for name in st.session_state.artifacts.keys():
        if st.button(f"📄 {name}", key=name):
            st.info("Artifact content is stored in memory.")

# --- 5. MAIN INTERFACE ---
if mode == "💬 Multi-Chat + RAG":
    selected_model = st.selectbox("Choose Brain", list(API_KEYS.keys()))
    
    # Display Chat History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask a question or discuss the uploaded file..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            res_text = ""
            
            # --- RAG Logic Integration ---
            if uploaded_file and uploaded_file.type == "application/pdf":
                with st.status("Reading PDF with RAG..."):
                    vs = process_pdf_rag(uploaded_file)
                    search_results = vs.similarity_search(prompt, k=3)
                    context = "\n".join([d.page_content for d in search_results])
                    prompt = f"Using this context:\n{context}\n\nQuestion: {prompt}"

            # --- API Routing ---
            try:
                if "Gemini" in selected_model:
                    genai.configure(api_key=API_KEYS["gemini-3-flash-preview"])
                    model = genai.GenerativeModel('gemini-3-flash-preview') 
                    response = model.generate_content(prompt)
                    res_text = response.text

                elif "GPT" in selected_model:
                    client = openai.OpenAI(api_key=API_KEYS["GPT-4o-mini"])
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": prompt}]
                    )
                    res_text = response.choices[0].message.content

                elif "Claude" in selected_model:
                    client = anthropic.Anthropic(api_key=API_KEYS["Claude-3-Haiku"])
                    response = client.messages.create(
                        model="claude-3-haiku-20240307",
                        max_tokens=1024,
                        messages=[{"role": "user", "content": prompt}]
                    )
                    res_text = response.content[0].text

                st.markdown(res_text)
                st.session_state.messages.append({"role": "assistant", "content": res_text})
                
                # Auto-Artifact Creation (if code is detected)
                if "```" in res_text:
                    art_name = f"Code_{len(st.session_state.artifacts)+1}"
                    st.session_state.artifacts[art_name] = res_text

            except Exception as e:
                st.error(f"API Error: {str(e)}")

elif mode == "🎨 Image Gen":
    img_prompt = st.text_area("Describe the image you want to create...")
    if st.button("Generate Image"):
        st.info("Using Gemini 2.5 Flash Image Engine...")
        # Image generation logic call here...

elif mode == "🎬 Video Gen":
    vid_prompt = st.text_input("Describe the video scene...")
    if st.button("Generate Video"):
        st.info("Sending request to Magic Hour API...")
        # Video generation logic call here...
