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
 
## ==================== CONFIGURATION ====================
st.sidebar.header("🔧 Configuration")
st.sidebar.markdown("### API Keys (Add yours)")
openai_key = st.sidebar.text_input("OpenAI API Key", type="password")
gemini_key = st.sidebar.text_input("Gemini API Key", type="password")
anthropic_key = st.sidebar.text_input("Anthropic (Claude) API Key", type="password")
groq_key = st.sidebar.text_input("Groq API Key", type="password")

# Model selector
selected_model = st.selectbox(
    "Choose LLM:",
    ["General Assistant", "Document Q&A", " Code Helper", "Medical Info","Legal Assistant"," Finance Advisor","Creative Writer" ],
    help="Select which model to use for responses"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 Documentation")
st.sidebar.markdown("""
## Multi-LLM Chat App

### Supported Models:
- **OpenAI**: GPT-4o-mini (`gpt-4o-mini`)
- **Gemini**: gemini-1.5-flash
- **Groq**: llama3-8b-8192 (fastest)
- **Claude**: claude-3-5-sonnet-20240620

### Features:
- ✅ **Conversation History** - Model remembers full chat
- ✅ **Match-case routing** - Clean model switching
- ✅ **Session persistence** - History saved across refreshes
- ✅ **Error handling** - Graceful API failures

### Setup:
1. Add your API keys in sidebar
2. Select model
3. Start chatting!

**History format**: `st.session_state.messages[model_name]` stores role/content pairs
""")

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

 
