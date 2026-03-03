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
 
st.sidebar.header("🔧 Configuration")

# API Keys - Individual lines (CRITICAL FIX)
openai_key = st.sidebar.text_input("🔑 OpenAI", type="password")
gemini_key = st.sidebar.text_input("🔑 Gemini", type="password")
anthropic_key = st.sidebar.text_input("🔑 Claude", type="password")
groq_key = st.sidebar.text_input("🔑 Groq", type="password")

st.sidebar.markdown("---")

# Model selector - Clean list (NO assignments inside)
selected_model = st.sidebar.selectbox(
    "Choose LLM:", 
    ["gpt-4o-mini", "gemini-pro", "claude-3-5-sonnet-20241022", "llama3-groq-70b-8192-tool-use-preview"]
)

 

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

 
