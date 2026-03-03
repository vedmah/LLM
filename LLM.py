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
 
st.subheader("🤖 LLM Selection")
    provider_options = ["OpenAI", "Gemini", "Claude", "Groq"]
    provider = st.selectbox("Provider:", provider_options, key="provider")
    
    model_options = {
        "OpenAI": ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
        "Gemini": ["gemini-1.5-flash", "gemini-1.5-pro"],
        "Claude": ["claude-3-5-sonnet-20241022", "claude-3-opus-20240229"],
        "Groq": ["llama-3.1-70b-versatile", "mixtral-8x7b-32768"]
    }
    selected_model = st.selectbox("Model:", model_options[provider], key="model")

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

 
