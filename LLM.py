import streamlit as st
from openai import OpenAI
import google.generativeai as genai
from anthropic import Anthropic
import os
from typing import List, Dict
import json

# Page config
st.set_page_config(page_title="Multi-LLM Chat", layout="wide", page_icon="🤖")

# Custom CSS
st.markdown("""
<style>
.chat-model-badge {
    background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 0.5rem 1rem;
    border-radius: 25px;
    font-weight: bold;
    font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)

# Hero Section
st.title("Multi-LLM Chat")
        
    
# Model selector - Clean list (NO assignments inside)
st.sidebar.header("Configuration") 
selected_model = st.sidebar.selectbox(
    "Choose LLM:", 
    ["gpt-4o-mini", "gemini-pro", "claude-3-5-sonnet-20241022", "llama3-groq-70b-8192-tool-use-preview"]
)

st.sidebar.header("🎭 Chat Mode")
selected_model = st.sidebar.selectbox(
    "Choose Mode:",
    [" General Assistant", "Code Helper 💻", "Document Q&A 📄", "Creative Writer ✍️", "Research Analyst 🔬"]       
)

st.sidebar.header("📁 Files Upload")

uploaded_files = st.sidebar.selectbox( 
        "Upload files (PDF, Images, TXT, ZIP for folders)", 
        accept_multiple_files=True,
        type=['pdf', 'png', 'jpg', 'jpeg', 'txt', 'zip']
    )
    
if uploaded_files:
        for file in uploaded_files:
            st.success(f"✅ {file.name} ({file.size/1024:.1f} KB)")
        st.session_state.uploaded_files = uploaded_files    
    
     

## ==================== CHAT HISTORY MANAGEMENT ====================
@st.cache_data
def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = {
            "openai": [],
            "gemini": [],
            "groq": [],
            "claude": []
        }

init_session_state()

 
