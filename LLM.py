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

# ==================== CHAT INPUT & AI RESPONSE ====================
if prompt := st.chat_input("💭 Ask anything about your files or general questions..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with chat_container:
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate AI response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.info("🤔 AI is thinking...")
            
            # Mode-based system prompts
            mode_prompts = {
                "General Assistant": "You are a helpful AI assistant.",
                "Code Helper 💻": "You are an expert programmer. Provide complete working code with explanations.",
                "Document Q&A 📄": "You are a document analysis expert. Reference uploaded files when relevant.",
                "Creative Writer ✍️": "You are a creative writing assistant.",
                "Research Analyst 🔬": "You are a research expert with detailed analysis."
            }
            
            system_prompt = mode_prompts.get(selected_mode, "You are a helpful assistant.")
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
            
            # Get API keys from session state
            openai_key = st.session_state.get("openai_key", "")
            gemini_key = st.session_state.get("gemini_key", "")
            claude_key = st.session_state.get("claude_key", "")
            groq_key = st.session_state.get("groq_key", "")
            
            # Extract model name and provider
            model_name = selected_llm.split(" (")[0]
            provider = selected_llm.split(" (")[1].replace(")", "") if "(" in selected_llm else "OpenAI"
            
            try:
                if "OpenAI" in selected_llm and openai_key:
                    client = OpenAI(api_key=openai_key)
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        temperature=0.7,
                        max_tokens=1500
                    )
                    full_response = response.choices[0].message.content
                
                elif "Gemini" in selected_llm and gemini_key:
                    genai.configure(api_key=gemini_key)
                    model_obj = genai.GenerativeModel(model_name)
                    response = model_obj.generate_content(prompt)
                    full_response = response.text
                
                elif "Claude" in selected_llm and claude_key:
                    client = Anthropic(api_key=claude_key)
                    response = client.messages.create(
                        model=model_name,
                        max_tokens=1500,
                        temperature=0.7,
                        messages=[{"role": "user", "content": prompt}]
                    )
                    full_response = response.content[0].text
                
                elif "Groq" in selected_llm and groq_key:
                    client = Groq(api_key=groq_key)
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        temperature=0.7,
                        max_tokens=1500
                    )
                    full_response = response.choices[0].message.content
                
                else:
                    full_response = "❌ Please add the API key for your selected LLM in the Configuration section."
            
            except Exception as e:
                full_response = f"⚠️ Error: {str(e)}"
            
            # Display response
            message_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})

# Sidebar clear button
if st.sidebar.button("🗑️ Clear Chat", use_container_width=True):
    st.session_state.messages = []
    st.rerun()

# Footer
st.markdown("---")
st.markdown("*Multi-LLM Chat - Powered by OpenAI, Gemini, Claude & Groq*")
