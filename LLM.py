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
uploaded_files = st.sidebar.file_uploader(
    " Upload Files",  
    type=["pdf", "png", "jpg", "jpeg", "txt", "zip"],
    accept_multiple_files=True
)


if uploaded_files:
        for file in uploaded_files:
            st.success(f"✅ {file.name} ({file.size/1024:.1f} KB)")
        st.session_state.uploaded_files = uploaded_files

st.subheader("🔍 Custom Prompt")
custom_prompt = st.text_area( 
        "Enter custom system prompt :",
        placeholder="e.g., 'You are a Python expert who explains code clearly...'",
        height=100,
        key="custom_prompt"
) 
if 'uploaded_files' in st.session_state:
        st.info(f"📊 {len(st.session_state.uploaded_files)} files ready for analysis")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Chat container
chat_container = st.container()

with chat_container:
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Regenerate button for assistant messages
            if message["role"] == "assistant":
                if st.button("🔄 Regenerate", key=f"regen_{len(st.session_state.messages)}"):
                    st.session_state.messages.pop()
                    st.rerun()    
    
if prompt := st.chat_input("💭 Ask anything about your files or general questions..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with chat_container:
        with st.chat_message("user"):
            st.markdown(prompt)

    # Generate response
    with chat_container:
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = generate_response(
                prompt, selected_model, provider, 
                openai_key, gemini_key, claude_key, groq_key,
                selected_mode, custom_prompt, temperature, max_tokens
            )
            message_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})         
    

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

 
