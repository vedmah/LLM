import streamlit as st
from openai import OpenAI
import google.generativeai as genai
from anthropic import Anthropic
from groq import Groq
import os

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
st.title("🤖 Multi-LLM Chat")

# ==================== SESSION STATE INITIALIZATION ====================
if "messages" not in st.session_state:
    st.session_state.messages = []

# ==================== SIDEBAR CONFIGURATION ====================
st.sidebar.header("🔧 Configuration")   
selected_llm = st.sidebar.selectbox(
    "🤖 Choose LLM:", 
    ["gpt-4o-mini (OpenAI)", "gemini-1.5-flash (Gemini)", "claude-3-5-sonnet-20241022 (Claude)", "llama3-8b-8192 (Groq)"]
)

# Chat Mode selector - FIXED variable name conflict
selected_mode = st.sidebar.selectbox(
    "🎭 Choose Mode:",
    ["General Assistant", "Code Helper 💻", "Document Q&A 📄", "Creative Writer ✍️", "Research Analyst 🔬"]
)

st.sidebar.divider()

# File Upload - FIXED
st.sidebar.header("📁 Files")
uploaded_files = st.sidebar.file_uploader(
    "Upload Files (PDF, Images, TXT, ZIP)",  
    type=["pdf", "png", "jpg", "jpeg", "txt", "zip"],
    accept_multiple_files=True
)

if uploaded_files:
    for file in uploaded_files:
        st.sidebar.success(f"✅ {file.name} ({file.size/1024:.1f} KB)")
    st.session_state.uploaded_files = uploaded_files
else:
    st.sidebar.info("👆 Upload files for analysis")

# ==================== CHAT DISPLAY ====================
chat_container = st.container()

with chat_container:
    # Display chat history
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Regenerate button - FIXED key
            if message["role"] == "assistant":
                if st.button("🔄 Regenerate", key=f"regen_{i}"):
                    st.session_state.messages = st.session_state.messages[:i]
                    st.rerun()

# ==================== CHAT INTERFACE ====================
chat_container = st.container()

with chat_container:
    # Show chat history
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            if message["role"] == "assistant":
                col1, col2 = st.columns([3, 1])
                with col2:
                    if st.button("🔄 Regenerate", key=f"regen_{i}", use_container_width=True):
                        st.session_state.messages = st.session_state.messages[:i]
                        st.rerun()

# Chat input
if prompt := st.chat_input("💭 Ask me anything... (No API keys needed!)"):
    # User message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with chat_container:
        with st.chat_message("user"):
            st.markdown(prompt)

        # AI Response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.info("🤔 Thinking...")
            
            # Generate FREE builtin response
            files = st.session_state.get("uploaded_files", [])
            full_response = generate_free_response(prompt, selected_llm, selected_mode, files)
            
            message_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #64748b; font-size: 0.9rem;'>
    🚀 Instant AI responses • No setup required • Professional-grade personalities
</div>
""", unsafe_allow_html=True)
# Footer
st.markdown("---")
st.markdown("*Multi-LLM Chat - Powered by OpenAI, Gemini, Claude & Groq*")
