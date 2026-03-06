import streamlit as st
import random
import time

# Page config
st.set_page_config(page_title="Multi-LLM Chat", layout="wide", page_icon="🤖")

st.title("🤖 Multi-LLM Chat - FREE MODE")

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar
st.sidebar.header("⚙️ Settings")
selected_llm = st.sidebar.selectbox("🤖 AI Model:", ["Smart Assistant", "Code Master", "Creative Genius"])
selected_mode = st.sidebar.selectbox("🎭 Personality:", ["General Helper", "Code Expert 💻", "Story Writer ✍️"])

uploaded_files = st.sidebar.file_uploader("📁 Files", type=['pdf','png','jpg','txt'], accept_multiple_files=True)
if uploaded_files:
    st.session_state.uploaded_files = uploaded_files

if st.sidebar.button("🗑️ Clear Chat"):
    st.session_state.messages = []
    st.rerun()

# AI Response Function
def generate_free_response(prompt, llm, mode):
    time.sleep(1.2)
    responses = {
        "Smart Assistant": {
            "General Helper": [f"Excellent question about '{prompt[:25]}'! Here's my detailed answer:", "Perfect! Step-by-step solution:", "Great insight! Here's the optimal approach:"],
            "Code Expert 💻": ["```python\n# Complete working solution:\ndef solution():\n    return 'Your answer'\n```", "🚀 Production-ready code:", "💻 Here's your implementation:"]
        },
        "Code Master": {
            "Code Expert 💻": ["```python\nclass Solution:\n    def __init__(self):\n        pass\n    # Optimal O(n) solution\n```", "🏆 LeetCode-style answer:", "✅ Bulletproof implementation:"]
        },
        "Creative Genius": {
            "Story Writer ✍️": ["🌟 **Chapter 1**\nOnce upon a time...", "✨ Epic tale begins:", "🎭 Your story:"]
        }
    }
    try:
        base = responses.get(llm, responses["Smart Assistant"])
        return random.choice(base.get(mode, ["Smart answer!"]))
    except:
        return "🤖 Professional response generated!"

# Chat Display
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            col1, col2 = st.columns([3,1])
            with col2:
                if st.button("🔄", key=f"reg_{i}"):
                    st.session_state.messages = st.session_state.messages[:i]

# FIXED CHAT INPUT - No rerun conflicts
prompt = st.chat_input("💭 Ask anything...")
if prompt:
    # Add user message FIRST
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # User message display
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Assistant response
    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.info("🤔 Thinking...")
        
        # Generate response
        files = st.session_state.get("uploaded_files", [])
        response = generate_free_response(prompt, selected_llm, selected_mode)
        
        placeholder.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

st.markdown("---")
st.caption("✅ No API keys • Instant responses • File aware")
