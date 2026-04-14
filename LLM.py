import streamlit as st
import openai
import anthropic
import google.generativeai as genai
import groq
import os
import json

# LangChain for RAG
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tempfile

# ========================================
# 🔥 BUILT-IN API KEYS - REPLACE THESE
# ========================================
API_KEYS = {
    "groq": "gsk_your_free_groq_key_here...",  # https://console.groq.com/keys (FREE)
    "google": "AIzaSy_your_free_gemini_key...",  # https://aistudio.google.com/app/apikey (FREE)
    "openai": "sk-proj_your_openai_key...",  # Optional
    "anthropic": "sk-ant-api03_your_claude_key..."  # Optional
}

FREE_MODELS = {
    "🚀 Groq Llama3-70B (FREE)": ("groq", "llama3-70b-8192"),
    "⚡ Gemini 1.5 Flash (FREE)": ("google", "gemini-1.5-flash-exp"),
    "⭐ GPT-4o-mini": ("openai", "gpt-4o-mini"),
    "🎯 Claude 3.5 Haiku": ("anthropic", "claude-3-5-haiku-latest")
}

# Pro CSS
st.set_page_config(page_title="🤖 Pro AI Chat", layout="wide", page_icon="🤖")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
.stApp { font-family: 'Inter', sans-serif; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
.main-header { text-align: center; padding: 2rem; color: white; }
.model-badge { 
    background: linear-gradient(45deg, #10b981, #059669); 
    color: white; padding: 0.6rem 1.2rem; border-radius: 25px; 
    font-weight: 600; font-size: 0.9rem; display: inline-block; margin: 1rem 0;
}
.chat-bubble { border-radius: 20px !important; padding: 1.2rem !important; margin: 1rem 0 !important; box-shadow: 0 4px 20px rgba(0,0,0,0.1); }
.user-bubble { background: linear-gradient(135deg, #3b82f6, #1d4ed8) !important; color: white !important; }
.assistant-bubble { background: rgba(255,255,255,0.95) !important; color: #111 !important; backdrop-filter: blur(20px); }
.btn-regen { background: #f59e0b !important; color: white !important; border-radius: 50% !important; width: 36px; height: 36px; border: none !important; }
.btn-regen:hover { background: #d97706 !important; transform: scale(1.1) !important; }
</style>
""", unsafe_allow_html=True)

# FIXED State - Single source of truth
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None

# Sidebar
with st.sidebar:
    st.markdown("### 🎛️ Controls")
    selected_model_display = st.selectbox("🤖 AI Model", list(FREE_MODELS.keys()), index=0)
    model_key, model_id = FREE_MODELS[selected_model_display]
    
    st.markdown("### 📄 RAG")
    uploaded_file = st.file_uploader("Upload PDF/TXT", type=['pdf','txt'])
    
    if uploaded_file and st.button("🔄 Process", type="primary"):
        with st.spinner("Processing..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name.split('.')[-1]) as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
            
            loader = PyPDFLoader(tmp_path) if uploaded_file.name.endswith('.pdf') else TextLoader(tmp_path)
            docs = loader.load()
            
            splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
            chunks = splitter.split_documents(docs)
            
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=API_KEYS["google"])
            st.session_state.vectorstore = FAISS.from_documents(chunks, embeddings)
            os.unlink(tmp_path)
            st.success("✅ RAG Ready!")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1: 
        if st.button("🗑️ Clear"): st.session_state.messages = []; st.session_state.vectorstore = None; st.rerun()
    with col2:
        if st.session_state.messages:
            st.download_button("💾 Export", json.dumps(st.session_state.messages, indent=2), "chat.json")

# Header
st.markdown(f"""
<div class="main-header">
    <h1 style="font-size: 3rem; margin: 0;">🤖 Pro AI Chat</h1>
    <p style="font-size: 1.3rem; opacity: 0.9;">FREE • RAG • Professional</p>
    <div class="model-badge">{selected_model_display}</div>
</div>
""", unsafe_allow_html=True)

# Chat History
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and st.button("🔄", key=f"regen_{i}", help="Regenerate"):
            st.session_state.messages = st.session_state.messages[:len(st.session_state.messages)-1]
            st.rerun()

# FIXED CHAT INPUT - No more errors!
if prompt := st.chat_input("💭 Ask anything..."):
    # STEP 1: Show user message immediately
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # STEP 2: Generate response
    with st.chat_message("assistant"):
        status = st.status("🤔 Thinking...", expanded=False)
        response_container = st.container()
        
        try:
            # RAG
            context = ""
            if st.session_state.vectorstore:
                docs = st.session_state.vectorstore.similarity_search(prompt, k=2)
                context = "\n\n📚 CONTEXT:\n" + "\n".join([doc.page_content[:400]+"..." for doc in docs])

            # History for context
            history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
            
            # FIXED GENERATION - Single response variable
            response_text = ""
            
            if model_key == "groq":
                client = groq.Groq(api_key=API_KEYS["groq"])
                stream = client.chat.completions.create(
                    model=model_id,
                    messages=history,
                    stream=True
                )
                for chunk in stream:
                    if chunk.choices[0].delta.content:
                        response_text += chunk.choices[0].delta.content
                        response_container.markdown(response_text + "▌")
            
            elif model_key == "google":
                genai.configure(api_key=API_KEYS["google"])
                model = genai.GenerativeModel(model_id)
                response = model.generate_content(prompt + context)
                response_text = response.text
            
            elif model_key == "openai":
                client = openai.OpenAI(api_key=API_KEYS["openai"])
                response = client.chat.completions.create(model=model_id, messages=history)
                response_text = response.choices[0].message.content
            
            else:  # claude
                client = anthropic.Anthropic(api_key=API_KEYS["anthropic"])
                msg = client.messages.create(model=model_id, max_tokens=1500, messages=[{"role": "user", "content": prompt + context}])
                response_text = msg.content[0].text
            
            status.update(label="✅ Complete!", state="complete")
            response_container.markdown(response_text)
            
        except Exception as e:
            status.update(label=f"❌ {str(e)}", state="error")
            response_container.error("Generation failed - check API keys")
        
        # STEP 3: FIXED - Save ONLY response_text
        st.session_state.messages.append({"role": "assistant", "content": response_text})

# Footer
st.markdown("<div style='text-align:center;padding:2rem;color:rgba(255,255,255,0.6);'>⚡ Pro AI • No config needed</div>", unsafe_allow_html=True)
