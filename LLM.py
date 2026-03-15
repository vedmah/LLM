import streamlit as st
from openai import OpenAI
import google.generativeai as genai
from anthropic import Anthropic
from groq import Groq
from perplexity import Perplexity  # pip install perplexity-ai (official from GitHub)
import os
from typing import Dict, List, Any
import json

# Page config
st.set_page_config(page_title="All-in-One LLM Chatbot", layout="wide", page_icon="🤖")

# Custom CSS for professional look
st.markdown("""
<style>
.chat-model-badge {
    background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 0.5rem 1rem;
    border-radius: 25px;
    font-weight: bold;
    font-size: 0.9rem;
    margin: 0.2rem;
}
.stChatMessage {
    padding: 1rem;
    border-radius: 15px;
    margin-bottom: 1rem;
}
.user-message { background: #e3f2fd; }
.assistant-message { background: #f5f5f5; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
@st.cache_data
def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = {
            "chatgpt": [],
            "claude": [],
            "perplexity": [],
            "gemini": [],
            "groq": []
        }
    if "current_model" not in st.session_state:
        st.session_state.current_model = "chatgpt"

init_session_state()

# Sidebar for API keys and model selection
with st.sidebar:
    st.header("🔑 API Keys")
    openai_key = st.text_input("OpenAI (ChatGPT)", type="password", key="openai_key")
    claude_key = st.text_input("Anthropic (Claude)", type="password", key="claude_key")
    perplexity_key = st.text_input("Perplexity", type="password", key="perplexity_key")
    gemini_key = st.text_input("Google Gemini", type="password", key="gemini_key")
    groq_key = st.text_input("Groq", type="password", key="groq_key")
    
    st.divider()
    st.header("🤖 Select Model")
    models = ["chatgpt", "claude", "perplexity", "gemini", "groq"]
    selected_model = st.selectbox("Choose LLM:", models, index=models.index(st.session_state.current_model))
    if selected_model != st.session_state.current_model:
        st.session_state.current_model = selected_model
        st.rerun()
    
    st.info("💡 Get keys:\n- OpenAI: platform.openai.com\n- Claude: console.anthropic.com\n- Perplexity: perplexity.ai/settings\n- Gemini: aistudio.google.com\n- Groq: console.groq.com")

# Model clients (lazy init)
def get_client(model: str) -> Any:
    if model == "chatgpt":
        if not openai_key: return None
        return OpenAI(api_key=openai_key)
    elif model == "claude":
        if not claude_key: return None
        return Anthropic(api_key=claude_key)
    elif model == "perplexity":
        if not perplexity_key: return None
        return Perplexity(api_key=perplexity_key)
    elif model == "gemini":
        if not gemini_key: return None
        genai.configure(api_key=gemini_key)
        return genai.GenerativeModel('gemini-1.5-pro')
    elif model == "groq":
        if not groq_key: return None
        return Groq(api_key=groq_key)
    return None

# Generate response based on model
async def generate_response(model_name: str, messages: List[Dict]) -> str:
    client = get_client(model_name)
    if not client:
        return "❌ API key missing for this model. Add it in sidebar."
    
    try:
        if model_name == "chatgpt":
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": m["role"], "content": m["content"]} for m in messages[-10:]],
                max_tokens=1000
            )
            return response.choices[0].message.content
        elif model_name == "claude":
            msg_history = [{"role": m["role"], "content": m["content"]} for m in messages[-10:]]
            response = client.messages.create(
                model="claude-3-5-sonnet-20240620",
                max_tokens=1000,
                messages=msg_history
            )
            return response.content[0].text
        elif model_name == "perplexity":
            response = client.chat.create(
                model="llama-3.1-sonar-large-128k-online",
                messages=[{"role": m["role"], "content": m["content"]} for m in messages[-5:]]
            )
            return response.choices[0].message.content
        elif model_name == "gemini":
            history = ""
            for m in messages[-10:]:
                history += f"{m['role']}: {m['content']}\n"
            response = client.generate_content(history)
            return response.text
        elif model_name == "groq":
            response = client.chat.completions.create(
                model="llama-3.1-70b-versatile",
                messages=[{"role": m["role"], "content": m["content"]} for m in messages[-10:]],
                max_tokens=1000
            )
            return response.choices[0].message.content
    except Exception as e:
        return f"❌ Error: {str(e)}"

# Main chat interface
st.title("🤖 All-in-One LLM Chatbot")
st.markdown(f"**Current Model:** <span class='chat-model-badge'>{st.session_state.current_model.upper()}</span>", unsafe_allow_html=True)

# Display chat history
current_messages = st.session_state.messages[st.session_state.current_model]
for i, msg in enumerate(current_messages):
    with st.chat_message(msg["role"], key=f"msg_{i}"):
        st.markdown(msg["content"], unsafe_allow_html=True)
        if msg["role"] == "assistant":
            st.button("🔄 Regenerate", on_click=lambda: regenerate(i), key=f"regen_{i}")

# Chat input
if prompt := st.chat_input("Ask anything..."):
    # Add user message
    current_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_response(st.session_state.current_model, current_messages)
            st.markdown(response)
    
    current_messages.append({"role": "assistant", "content": response})
    st.rerun()

# Clear chat button
if st.button("🗑️ Clear Chat History", type="secondary"):
    st.session_state.messages[st.session_state.current_model] = []
    st.rerun()

# Footer
st.markdown("---")
st.caption("Built for multi-LLM switching with persistent history. Deploy: `streamlit run multi_llm_chat.py` 🚀")
