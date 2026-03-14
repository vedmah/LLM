import streamlit as st
import os
from typing import List, Dict
import json
import tempfile
from langchain_community.document_loaders import PyPDFLoader, UnstructuredLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_chroma import Chroma
from langchain.schema import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
import tiktoken  # For token counting

# Page config
st.set_page_config(page_title="Multi-LLM Chat", layout="wide", page_icon="🤖")

# Custom CSS
st.markdown("""
<style>
.chat-model-badge {
    background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
    color: white; padding: 0.5rem 1rem; border-radius: 25px;
    font-weight: bold; font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)

# API Keys (use secrets.toml or env vars in production)
openai_key = st.sidebar.text_input("OpenAI API Key", type="password")
gemini_key = st.sidebar.text_input("Gemini API Key", type="password")
claude_key = st.sidebar.text_input("Anthropic API Key", type="password")
groq_key = st.sidebar.text_input("Groq API Key", type="password")  # Assume for Llama via Groq

# Model selector
st.sidebar.header("🤖 LLM")
selected_model = st.sidebar.selectbox(
    "Choose LLM:",
    ["gpt-4o-mini", "gemini-pro", "claude-3-5-sonnet-20241022", "llama3-groq-70b-8192-tool-use-preview"]
)

# Chat Mode
st.sidebar.header("🎭 Chat Mode")
selected_mode = st.sidebar.selectbox(
    "Choose Mode:",
    ["General Assistant", "Code Helper 💻", "Document Q&A 📄", "Creative Writer ✍️", "Research Analyst 🔬"]
)

# Files Upload for RAG
st.sidebar.header("📁 Files Upload")
uploaded_files = st.sidebar.file_uploader(
    "Upload Files", type=["pdf", "png", "jpg", "jpeg", "txt"], accept_multiple_files=True
)

# Initialize session state
@st.cache_resource
def init_session():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = {}

init_session()

# Process uploaded files for RAG
if uploaded_files:
    if st.sidebar.button("Process Files for RAG"):
        docs = []
        for file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as tmp:
                tmp.write(file.getvalue())
                tmp_path = tmp.name
            if file.name.endswith('.pdf'):
                loader = PyPDFLoader(tmp_path)
            else:
                loader = UnstructuredLoader(tmp_path)  # Handles images/txt
            docs.extend(loader.load())
            os.unlink(tmp_path)
        
        # Chunk and embed
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)
        embeddings = OpenAIEmbeddings(openai_api_key=openai_key) if openai_key else None
        
        if embeddings:
            st.session_state.vectorstore = Chroma.from_documents(
                chunks, embeddings, persist_directory="./chroma_db"
            )
            st.sidebar.success("✅ Files processed and indexed for RAG!")
        else:
            st.sidebar.error("❌ OpenAI key needed for embeddings.")

# Get LLM instance
@st.cache_resource
def get_llm(model_name: str):
    if "gpt" in model_name:
        return ChatOpenAI(model=model_name, api_key=openai_key, temperature=0.7)
    elif "gemini" in model_name:
        return ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=gemini_key, temperature=0.7)
    elif "claude" in model_name:
        return ChatAnthropic(model=model_name, api_key=claude_key, temperature=0.7)
    elif "llama3-groq" in model_name:
        return ChatOpenAI(model="llama3-groq-70b-8192-tool-use-preview", api_key=groq_key, temperature=0.7)
    return None

llm = get_llm(selected_model)

# RAG Chain if Document Q&A and vectorstore exists
if selected_mode == "Document Q&A 📄" and st.session_state.vectorstore and llm:
    retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 4})
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant. Answer using ONLY the provided context from documents.
Context: {context}
Mode: {mode}"""),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])
    
    chain = {
        "input": lambda x: x,
        "context": retriever,
        "mode": lambda x: selected_mode,
        "history": lambda x: st.session_state.chat_history.get(st.session_state.get("session_id", "default"), [])
    } | prompt | llm | StrOutputParser()

# General response generator (non-RAG)
def generate_general_response(prompt: str):
    if not llm:
        return "Please set API keys and select a model."
    
    custom_prompt = f"Mode: {selected_mode}. Respond to: {prompt}"
    return llm.invoke(custom_prompt).content

# Chat UI
st.title("Multi-LLM Chat")
chat_container = st.container()

with chat_container:
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and st.button("🔄 Regenerate", key=f"regen_{i}"):
                st.session_state.messages = st.session_state.messages[:i]
                st.rerun()

if prompt := st.chat_input("💭 Ask anything about your files or general questions..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with chat_container:
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            if selected_mode == "Document Q&A 📄" and st.session_state.vectorstore:
                response = chain.invoke(prompt)
            else:
                response = generate_general_response(prompt)
            message_placeholder.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

# Clear chat
if st.sidebar.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun() 
