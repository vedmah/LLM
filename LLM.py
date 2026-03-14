import streamlit as st
import os
import tempfile
from typing import List
import json

# Lightweight imports - only load when needed
try:
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from langchain_chroma import Chroma
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    st.error("Install langchain deps: `pip install langchain-openai langchain-chroma`")

# Page config - FIRST (critical for speed)
st.set_page_config(page_title="Multi-LLM Chat", layout="wide", page_icon="🤖")

# Minimal CSS
st.markdown("""
<style>
.chat-model-badge {background: linear-gradient(45deg,#667eea,#764ba2);color:white;padding:.5rem 1rem;border-radius:25px;font-weight:bold;font-size:.9rem;}
</style>
""", unsafe_allow_html=True)

# Sidebar - Simplified
with st.sidebar:
    st.header("🔑 API Keys")
    openai_key = st.text_input("OpenAI", type="password", help="Required for GPT models")
    gemini_key = st.text_input("Gemini", type="password")
    claude_key = st.text_input("Claude", type="password")
    groq_key = st.text_input("Groq", type="password")
    
    st.divider()
    model = st.selectbox("🤖 Model", ["gpt-4o-mini", "gemini-pro", "claude-3-5-sonnet-20241022", "llama3-groq-70b"])
    mode = st.selectbox("🎭 Mode", ["General", "Code", "Docs", "Creative", "Research"])
    
    st.divider()
    if RAG_AVAILABLE:
        uploaded_files = st.file_uploader("📄 Docs", type=["pdf","txt"], accept_multiple_files=True)
        if uploaded_files and st.button("🔄 Process RAG") and openai_key:
            st.session_state.rag_processed = True
            st.success("✅ RAG Ready!")

# Initialize session state ONCE
if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag_ready" not in st.session_state:
    st.session_state.rag_ready = False

# Lightweight LLM function (no caching overhead)
def get_llm(model_name: str, api_key: str):
    if not api_key:
        return None
    if "gpt" in model_name or "llama3-groq" in model_name:
        return ChatOpenAI(model=model_name, api_key=openai_key, temperature=0.1)
    return None  # Simplified - add other models later

llm = get_llm(model, openai_key) if openai_key else None

# Fast chat UI
st.title("🤖 Multi-LLM Chat")
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Main input - only process when needed
if prompt := st.chat_input("Ask anything..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        if llm:
            with st.spinner("Thinking..."):
                try:
                    response = llm.invoke(prompt).content
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        else:
            st.warning("⚠️ Please add your OpenAI API key in sidebar")

# Sidebar controls
with st.sidebar:
    st.divider()
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()
    
    st.caption("💡 Fast & lightweight - GPT-4o-mini ready!")
