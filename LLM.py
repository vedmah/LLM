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
selected_model = st.selectbox(
    "Choose API:",
    openai_key = st.sidebar.text_input("OpenAI API Key", type="password"),
    gemini_key = st.sidebar.text_input("Gemini API Key", type="password"),
    anthropic_key = st.sidebar.text_input("Anthropic (Claude) API Key", type="password"),
    groq_key = st.sidebar.text_input("Groq API Key", type="password"), 
    help="Select which model to use for responses"
)

# Model selector
selected_model = st.selectbox(
    "Choose LLM:",
    ["General Assistant", "Document Q&A", " Code Helper", "Medical Info","Legal Assistant"," Finance Advisor","Creative Writer" ],
    help="Select which model to use for responses"
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

 
