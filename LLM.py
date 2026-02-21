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
    ["openai", "gemini", "groq", "claude"],
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

# Get current model history
current_history: List[Dict[str, str]] = st.session_state.messages[selected_model]

## ==================== LLM CLIENTS ====================
def get_openai_client():
    if not openai_key:
        raise ValueError("OpenAI API key required")
    return OpenAI(api_key=openai_key)

def get_gemini_client():
    if not gemini_key:
        raise ValueError("Gemini API key required")
    genai.configure(api_key=gemini_key)
    return genai.GenerativeModel('gemini-1.5-flash')

def get_groq_client():
    if not groq_key:
        raise ValueError("Groq API key required")
    from groq import Groq
    return Groq(api_key=groq_key)

def get_claude_client():
    if not anthropic_key:
        raise ValueError("Anthropic API key required")
    return Anthropic(api_key=anthropic_key)

## ==================== GENERATE RESPONSE ====================
async def generate_response(model_name: str, prompt: str) -> str:
    try:
        match model_name:
            case "openai":
                client = get_openai_client()
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=current_history + [{"role": "user", "content": prompt}],
                    max_tokens=1000,
                    temperature=0.7
                )
                return response.choices[0].message.content

            case "gemini":
                client = get_gemini_client()
                history_formatted = [{"role": msg["role"], "parts": [msg["content"]]} for msg in current_history]
                history_formatted.append({"role": "user", "parts": [prompt]})
                
                response = client.generate_content(history_formatted)
                return response.text

            case "groq":
                client = get_groq_client()
                response = client.chat.completions.create(
                    model="llama3-8b-8192",
                    messages=current_history + [{"role": "user", "content": prompt}],
                    max_tokens=1000,
                    temperature=0.7
                )
                return response.choices[0].message.content

            case "claude":
                client = get_claude_client()
                messages_formatted = [{"role": msg["role"], "content": msg["content"]} for msg in current_history]
                messages_formatted.append({"role": "user", "content": prompt})
                
                response = client.messages.create(
                    model="claude-3-5-sonnet-20240620",
                    max_tokens=1000,
                    messages=messages_formatted
                )
                return response.content[0].text

    except Exception as e:
        return f"❌ Error with {model_name.upper()}: {str(e)}"

## ==================== MAIN UI ====================
st.title("🤖 Multi-LLM Chat")
st.markdown(f"**Selected Model:** <span class='chat-model-badge'>{selected_model.upper()}</span>", unsafe_allow_html=True)

# Chat display
chat_container = st.container()
with chat_container:
    if current_history:
        for message in current_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    else:
        st.info("👋 Ask me anything! History maintained per model.")

# Chat input
if prompt := st.chat_input("Type your message..."):
    # Add user message
    st.chat_message("user").markdown(prompt)
    current_history.append({"role": "user", "content": prompt})
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner(f"🤖 {selected_model.upper()} thinking..."):
            response = generate_response(selected_model, prompt)
            st.markdown(response)
    
    # Add to history
    current_history.append({"role": "assistant", "content": response})
    st.session_state.messages[selected_model] = current_history

# Clear chat button
if st.button("🗑️ Clear Chat History", type="secondary"):
    st.session_state.messages[selected_model] = []
    st.rerun()

# Export history
if st.sidebar.button("💾 Export Chat History"):
    st.sidebar.download_button(
        label="Download JSON",
        data=json.dumps(st.session_state.messages, indent=2),
        file_name="chat_history.json",
        mime="application/json"
    )
