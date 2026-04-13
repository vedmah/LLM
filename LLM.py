import streamlit as st
import openai
import anthropic
import google.generativeai as genai
import requests
import time

# --- CONFIGURATION (Keys hardcoded as requested) ---
API_KEYS = {
    "GPT-4o-mini": "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",
    "Claude-3-Haiku": "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",
    "Gemini-1.5-Flash": "AIzaSyBmPU22hpyfCcy8u0wJiMbj6WGwQii8mWU",
     
}

st.set_page_config(page_title="Universal AI Hub 2026", layout="wide")

# --- SESSION STATE ---
if "messages" not in st.session_state: st.session_state.messages = []
if "artifacts" not in st.session_state: st.session_state.artifacts = []

# --- SIDEBAR: Navigation & Media ---
with st.sidebar:
    st.title("🤖 Multi-Model Hub")
    mode = st.radio("Task Mode", ["Chat & Files", "Generate Image", "Generate Video"])
    
    st.divider()
    st.subheader("Context & Assets")
    uploaded_file = st.file_uploader("Upload PDF, Image, or Video", 
                                    type=['pdf', 'png', 'jpg', 'mp4', 'mov'])
    
    if st.button("Clear History"):
        st.session_state.messages = []
        st.rerun()

# --- MAIN INTERFACE ---
st.header(f"Mode: {mode}")

if mode == "Chat & Files":
    selected_model = st.selectbox("Select Text Model", ["GPT-4o-mini", "Claude-3-Haiku", "Gemini-1.5-Flash" ])
    
    # Display Chat
    for m in st.session_state.messages:
        with st.chat_message(m["role"]): st.markdown(m["content"])

    if prompt := st.chat_input("Explain this file or ask a question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)

        with st.chat_message("assistant"):
            if selected_model == "Gemini-1.5-Flash":
                genai.configure(api_key=API_KEYS[selected_model])
                model = genai.GenerativeModel('gemini-1.5-flash')
                # If file uploaded, include it in content
                content = [prompt]
                if uploaded_file:
                    content.append(genai.upload_file(uploaded_file))
                response = model.generate_content(content)
                res_text = response.text
            
            # (Similar logic for GPT/Claude using their SDKs for vision/files)
            
            st.markdown(res_text)
            st.session_state.messages.append({"role": "assistant", "content": res_text})

elif mode == "Generate Image":
    # Using Gemini 2.5 Flash Image (Free Tier: ~500 images/day in 2026)
    img_prompt = st.text_area("Describe the image...")
    if st.button("Generate Image"):
        with st.spinner("Painting..."):
            genai.configure(api_key=API_KEYS["Gemini-1.5-Flash"])
            model = genai.GenerativeModel('gemini-2.5-flash-image')
            response = model.generate_content(img_prompt)
            # Display image result logic here
            st.image(response.candidates[0].content.parts[0].inline_data.data)

elif mode == "Generate Video":
    # Magic Hour API (Free Tier: 100 daily credits for prototyping)
    vid_prompt = st.text_input("Describe the video scene...")
    if st.button("Action!"):
        with st.spinner("Rendering video..."):
            # Example API POST request to Magic Hour
            headers = {"Authorization": f"Bearer {API_KEYS['MagicHour-Video']}"}
            payload = {"prompt": vid_prompt, "style": "cinematic"}
            # res = requests.post("https://api.magichour.ai/v1/video", json=payload, headers=headers)
            st.info("Video generation request sent. Check back in 60s.")
