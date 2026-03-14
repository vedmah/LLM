import streamlit as st
from openai import OpenAI
import google.generativeai as genai
from anthropic import Anthropic
import os
import json
import re
import io
import hashlib
from typing import List, Dict, Tuple, Optional

# ── Optional heavy deps (graceful fallback) ───────────────────────────────────
try:
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False

try:
    import fitz  # PyMuPDF
    PYMUPDF_OK = True
except ImportError:
    PYMUPDF_OK = False

try:
    from PIL import Image
    PIL_OK = True
except ImportError:
    PIL_OK = False

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Multi-LLM Chat",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styles ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #0d1117; color: #e6edf3; }

[data-testid="stSidebar"] {
    background: #161b22 !important;
    border-right: 1px solid #30363d;
}
[data-testid="stSidebar"] * { color: #c9d1d9 !important; }

h1 { 
    font-size: 1.9rem !important; 
    font-weight: 700 !important; 
    background: linear-gradient(90deg, #58a6ff, #bc8cff);
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
}

/* Provider badge */
.provider-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: linear-gradient(135deg, #1f2937, #111827);
    border: 1px solid #374151; border-radius: 20px;
    padding: 4px 12px; font-size: 0.78rem; font-weight: 600;
    color: #9ca3af; margin-bottom: 8px;
}

/* Metric pill */
.rag-pill {
    background: #0d4429; border: 1px solid #238636;
    color: #3fb950; border-radius: 12px;
    padding: 2px 10px; font-size: 0.72rem; font-weight: 600;
    display: inline-block; margin-left: 6px;
}

/* File chip */
.file-chip {
    display: inline-flex; align-items: center; gap: 6px;
    background: #161b22; border: 1px solid #30363d;
    border-radius: 8px; padding: 4px 10px;
    font-size: 0.78rem; color: #8b949e; margin: 2px;
}

/* Code font */
code { font-family: 'JetBrains Mono', monospace !important; }

/* Buttons */
.stButton > button {
    background: #21262d; border: 1px solid #30363d; color: #c9d1d9;
    border-radius: 8px; font-size: 0.82rem;
    transition: background .2s, border-color .2s;
}
.stButton > button:hover { background: #30363d; border-color: #58a6ff; color: #fff; }

/* Chat messages */
[data-testid="stChatMessage"] {
    background: #161b22 !important;
    border: 1px solid #21262d !important;
    border-radius: 12px !important;
    margin-bottom: 8px !important;
}

/* Input */
[data-testid="stChatInput"] textarea {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    color: #e6edf3 !important;
    border-radius: 10px !important;
}

/* Selectbox / text_input */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    background: #161b22 !important;
    border-color: #30363d !important;
    color: #e6edf3 !important;
}
input[type="password"], input[type="text"] {
    background: #161b22 !important;
    color: #e6edf3 !important;
}

/* Expander */
details summary { color: #8b949e !important; font-size: 0.85rem; }

div[data-testid="stExpander"] {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE INIT
# ══════════════════════════════════════════════════════════════════════════════
def init_state():
    defaults = {
        "messages": [],               # [{role, content, model, rag_used}]
        "rag_chunks": [],             # [{text, source, embedding?}]
        "uploaded_hashes": set(),
        "rag_enabled": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ══════════════════════════════════════════════════════════════════════════════
# DOCUMENT PROCESSING & RAG
# ══════════════════════════════════════════════════════════════════════════════
def chunk_text(text: str, chunk_size: int = 600, overlap: int = 80) -> List[str]:
    """Split text into overlapping chunks."""
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
        i += chunk_size - overlap
    return chunks


def extract_text_from_file(file) -> str:
    """Extract plain text from uploaded file."""
    name = file.name.lower()
    try:
        if name.endswith(".txt") or name.endswith(".md"):
            return file.read().decode("utf-8", errors="ignore")
        elif name.endswith(".pdf"):
            if PYMUPDF_OK:
                data = file.read()
                doc  = fitz.open(stream=data, filetype="pdf")
                return "\n".join(page.get_text() for page in doc)
            else:
                return f"[PDF: {file.name} — install PyMuPDF to extract text]"
        elif name.endswith(".json"):
            data = json.loads(file.read())
            return json.dumps(data, indent=2)
        elif name.endswith(".py") or name.endswith(".js") or name.endswith(".ts"):
            return file.read().decode("utf-8", errors="ignore")
        elif name.endswith((".png", ".jpg", ".jpeg")):
            return f"[Image file: {file.name} — use a vision-capable model for image analysis]"
        else:
            try:
                return file.read().decode("utf-8", errors="ignore")
            except Exception:
                return f"[Binary file: {file.name} — text extraction not supported]"
    except Exception as e:
        return f"[Error reading {file.name}: {e}]"


def simple_embed(text: str) -> List[float]:
    """Lightweight TF-IDF-style bag-of-words vector (no external API needed)."""
    words  = re.findall(r"\w+", text.lower())
    counts: Dict[str, int] = {}
    for w in words:
        counts[w] = counts.get(w, 0) + 1
    # Fixed vocab hash trick: 512 dimensions
    vec = [0.0] * 512
    for w, c in counts.items():
        idx = int(hashlib.md5(w.encode()).hexdigest(), 16) % 512
        vec[idx] += c
    norm = sum(v ** 2 for v in vec) ** 0.5 or 1.0
    return [v / norm for v in vec]


def process_files(files) -> int:
    """Process uploaded files into RAG chunks. Returns number of new chunks added."""
    added = 0
    for file in files:
        file.seek(0)
        fhash = hashlib.md5(file.read()).hexdigest()
        file.seek(0)
        if fhash in st.session_state.uploaded_hashes:
            continue
        st.session_state.uploaded_hashes.add(fhash)

        text   = extract_text_from_file(file)
        chunks = chunk_text(text)
        for chunk in chunks:
            entry = {"text": chunk, "source": file.name}
            if SKLEARN_OK:
                entry["embedding"] = simple_embed(chunk)
            st.session_state.rag_chunks.append(entry)
        added += len(chunks)
        file.seek(0)
    return added


def retrieve_context(query: str, top_k: int = 4) -> str:
    """Retrieve top-k relevant chunks for the query."""
    chunks = st.session_state.rag_chunks
    if not chunks:
        return ""

    if SKLEARN_OK and chunks[0].get("embedding"):
        q_vec   = simple_embed(query)
        scores  = cosine_similarity([q_vec], [c["embedding"] for c in chunks])[0]
        indices = scores.argsort()[::-1][:top_k]
    else:
        # Keyword fallback
        q_words = set(re.findall(r"\w+", query.lower()))
        scored  = []
        for i, c in enumerate(chunks):
            c_words = set(re.findall(r"\w+", c["text"].lower()))
            scored.append((len(q_words & c_words), i))
        scored.sort(reverse=True)
        indices = [i for _, i in scored[:top_k]]

    parts = []
    for i in indices:
        c = chunks[i]
        parts.append(f"[Source: {c['source']}]\n{c['text']}")
    return "\n\n---\n\n".join(parts)


# ══════════════════════════════════════════════════════════════════════════════
# LLM CLIENTS
# ══════════════════════════════════════════════════════════════════════════════
MODEL_PROVIDERS = {
    "gpt-4o":               "openai",
    "gpt-4o-mini":          "openai",
    "gpt-3.5-turbo":        "openai",
    "gemini-1.5-pro":       "gemini",
    "gemini-1.5-flash":     "gemini",
    "gemini-pro":           "gemini",
    "claude-3-5-sonnet-20241022": "claude",
    "claude-3-haiku-20240307":    "claude",
    "claude-3-opus-20240229":     "claude",
    "llama3-70b-8192":      "groq",
    "mixtral-8x7b-32768":   "groq",
    "llama3-8b-8192":       "groq",
}

MODE_PROMPTS = {
    "🤖 General Assistant":  "You are a helpful, concise, and friendly AI assistant.",
    "💻 Code Helper":        "You are an expert software engineer. Provide clean, well-commented code with explanations. Use markdown code blocks.",
    "📄 Document Q&A":       "You are a precise document analyst. Answer questions strictly based on provided context. Cite sources when possible.",
    "✍️ Creative Writer":    "You are a creative writer. Craft engaging, imaginative, and vivid content.",
    "🔬 Research Analyst":   "You are a rigorous research analyst. Provide structured, evidence-based analysis with citations.",
}


def call_openai(messages: List[Dict], model: str, api_key: str,
                temperature: float, max_tokens: int) -> str:
    client = OpenAI(api_key=api_key)
    resp   = client.chat.completions.create(
        model=model, messages=messages,
        temperature=temperature, max_tokens=max_tokens,
        stream=False,
    )
    return resp.choices[0].message.content


def call_gemini(messages: List[Dict], model: str, api_key: str,
                temperature: float, max_tokens: int) -> str:
    genai.configure(api_key=api_key)
    gmodel = genai.GenerativeModel(model)
    # Convert to Gemini format
    history, last = [], messages[-1]["content"]
    for m in messages[:-1]:
        role = "user" if m["role"] == "user" else "model"
        history.append({"role": role, "parts": [m["content"]]})
    chat = gmodel.start_chat(history=history)
    resp = chat.send_message(
        last,
        generation_config=genai.types.GenerationConfig(
            temperature=temperature, max_output_tokens=max_tokens
        ),
    )
    return resp.text


def call_claude(messages: List[Dict], system: str, model: str, api_key: str,
                temperature: float, max_tokens: int) -> str:
    client = Anthropic(api_key=api_key)
    resp   = client.messages.create(
        model=model, system=system, messages=messages,
        temperature=temperature, max_tokens=max_tokens,
    )
    return resp.content[0].text


def call_groq(messages: List[Dict], model: str, api_key: str,
              temperature: float, max_tokens: int) -> str:
    # Groq uses OpenAI-compatible endpoint
    client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
    resp   = client.chat.completions.create(
        model=model, messages=messages,
        temperature=temperature, max_tokens=max_tokens,
    )
    return resp.choices[0].message.content


def generate_response(
    user_prompt: str,
    model: str,
    mode: str,
    openai_key: str,
    gemini_key: str,
    claude_key: str,
    groq_key: str,
    temperature: float,
    max_tokens: int,
    rag_context: str = "",
) -> Tuple[str, bool]:
    """Returns (response_text, rag_was_used)."""
    provider   = MODEL_PROVIDERS.get(model, "openai")
    system_msg = MODE_PROMPTS.get(mode, MODE_PROMPTS["🤖 General Assistant"])
    rag_used   = bool(rag_context)

    if rag_used:
        system_msg += (
            "\n\nUse the following retrieved document context to answer the user's question. "
            "Reference the source when relevant.\n\n"
            f"=== RETRIEVED CONTEXT ===\n{rag_context}\n=== END CONTEXT ==="
        )

    # Build history from session (last 10 turns to stay within context)
    history = []
    for m in st.session_state.messages[-10:]:
        history.append({"role": m["role"], "content": m["content"]})
    history.append({"role": "user", "content": user_prompt})

    try:
        if provider == "openai":
            if not openai_key:
                return "⚠️ OpenAI API key not provided.", False
            msgs = [{"role": "system", "content": system_msg}] + history
            return call_openai(msgs, model, openai_key, temperature, max_tokens), rag_used

        elif provider == "gemini":
            if not gemini_key:
                return "⚠️ Gemini API key not provided.", False
            # Inject system as first user turn for Gemini
            msgs = [{"role": "user", "content": f"[System instruction]: {system_msg}\n\nAcknowledge briefly."},
                    {"role": "assistant", "content": "Understood."}] + history
            return call_gemini(msgs, model, gemini_key, temperature, max_tokens), rag_used

        elif provider == "claude":
            if not claude_key:
                return "⚠️ Claude API key not provided.", False
            return call_claude(history, system_msg, model, claude_key, temperature, max_tokens), rag_used

        elif provider == "groq":
            if not groq_key:
                return "⚠️ Groq API key not provided.", False
            msgs = [{"role": "system", "content": system_msg}] + history
            return call_groq(msgs, model, groq_key, temperature, max_tokens), rag_used

        else:
            return f"⚠️ Unknown provider for model: {model}", False

    except Exception as e:
        return f"❌ **Error calling {provider} API:**\n```\n{e}\n```", False


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🤖 Multi-LLM Chat")
    st.markdown("---")

    # ── Model selection ──
    st.markdown("### 🧠 Model")
    selected_model = st.selectbox(
        "LLM Model",
        list(MODEL_PROVIDERS.keys()),
        label_visibility="collapsed",
    )
    provider_label = MODEL_PROVIDERS.get(selected_model, "?").upper()
    st.markdown(f'<div class="provider-badge">🔌 Provider: {provider_label}</div>', unsafe_allow_html=True)

    # ── Mode ──
    st.markdown("### 🎭 Chat Mode")
    selected_mode = st.selectbox(
        "Mode",
        list(MODE_PROMPTS.keys()),
        label_visibility="collapsed",
    )

    # ── API keys ──
    st.markdown("### 🔑 API Keys")
    with st.expander("Configure Keys", expanded=False):
        openai_key = st.text_input("OpenAI Key",  type="password", placeholder="sk-...")
        gemini_key = st.text_input("Gemini Key",  type="password", placeholder="AIza...")
        claude_key = st.text_input("Claude Key",  type="password", placeholder="sk-ant-...")
        groq_key   = st.text_input("Groq Key",    type="password", placeholder="gsk_...")

    # ── Parameters ──
    st.markdown("### ⚙️ Parameters")
    temperature = st.slider("Temperature", 0.0, 1.5, 0.7, 0.05)
    max_tokens  = st.slider("Max Tokens",  128, 4096, 1024, 128)

    # ── File upload & RAG ──
    st.markdown("### 📁 Document RAG")
    uploaded_files = st.file_uploader(
        "Upload files",
        type=["pdf", "txt", "md", "py", "js", "ts", "json", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        with st.spinner("Indexing documents…"):
            new_chunks = process_files(uploaded_files)
        if new_chunks:
            st.success(f"✅ Indexed {new_chunks} new chunks")
        st.session_state.rag_enabled = True
        for f in uploaded_files:
            kb = f.size / 1024
            st.markdown(
                f'<div class="file-chip">📄 {f.name[:22]} <span style="color:#58a6ff">({kb:.1f} KB)</span></div>',
                unsafe_allow_html=True,
            )

    if st.session_state.rag_chunks:
        rag_on = st.toggle("Enable RAG retrieval", value=st.session_state.rag_enabled)
        st.session_state.rag_enabled = rag_on
        st.caption(f"📚 {len(st.session_state.rag_chunks)} chunks indexed across {len(uploaded_files or [])} file(s)")
        if st.button("🗑️ Clear documents"):
            st.session_state.rag_chunks = []
            st.session_state.uploaded_hashes = set()
            st.session_state.rag_enabled = False
            st.rerun()

    st.markdown("---")
    if st.button("🧹 Clear chat history"):
        st.session_state.messages = []
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PANEL
# ══════════════════════════════════════════════════════════════════════════════
# Header row
col_title, col_info = st.columns([3, 1])
with col_title:
    st.title("Multi-LLM Chat")
with col_info:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        f'<div style="text-align:right">'
        f'<span class="provider-badge">🤖 {selected_model}</span><br>'
        f'<span style="font-size:0.75rem;color:#6e7681">Mode: {selected_mode}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

# Welcome message
if not st.session_state.messages:
    st.markdown("""
    <div style="background:#161b22;border:1px solid #21262d;border-radius:12px;padding:1.5rem;margin:1rem 0;">
        <h3 style="color:#58a6ff;margin:0 0 0.5rem">👋 Welcome!</h3>
        <p style="color:#8b949e;margin:0">
            Choose a model & API key in the sidebar, then start chatting. 
            Upload documents to enable <strong style="color:#3fb950">RAG-powered Q&A</strong> over your files.
        </p>
    </div>
    """, unsafe_allow_html=True)

# Chat history
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            meta_parts = []
            if message.get("model"):
                meta_parts.append(f"🤖 `{message['model']}`")
            if message.get("rag_used"):
                meta_parts.append('<span class="rag-pill">RAG</span>')
            if meta_parts:
                st.markdown(
                    '<span style="font-size:0.72rem;color:#6e7681">' + " &nbsp;·&nbsp; ".join(meta_parts) + "</span>",
                    unsafe_allow_html=True,
                )
            # Regenerate
            if st.button("🔄", key=f"regen_{i}", help="Regenerate response"):
                # Remove last assistant message and re-submit last user prompt
                if len(st.session_state.messages) >= 2:
                    st.session_state.messages.pop()  # remove assistant
                    last_user = st.session_state.messages[-1]["content"]
                    st.session_state.messages.pop()  # remove user too (will re-add)
                    # re-trigger via user prompt mechanism
                    st.session_state["_reprompt"] = last_user
                st.rerun()

# Handle regenerate
reprompt = st.session_state.pop("_reprompt", None)

# Chat input
prompt = st.chat_input("💭 Ask anything… (RAG active if documents uploaded)")

if prompt or reprompt:
    user_input = prompt or reprompt

    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # RAG retrieval
    rag_context = ""
    if st.session_state.rag_enabled and st.session_state.rag_chunks:
        rag_context = retrieve_context(user_input, top_k=4)

    # Show retrieved context (collapsed)
    if rag_context:
        with st.expander("📎 Retrieved context (RAG)", expanded=False):
            st.markdown(f"```\n{rag_context[:1200]}{'...' if len(rag_context) > 1200 else ''}\n```")

    # Generate & stream response
    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("⏳ *Thinking…*")

        response, rag_used = generate_response(
            user_input,
            selected_model,
            selected_mode,
            openai_key  if "openai_key"  in dir() else "",
            gemini_key  if "gemini_key"  in dir() else "",
            claude_key  if "claude_key"  in dir() else "",
            groq_key    if "groq_key"    in dir() else "",
            temperature,
            max_tokens,
            rag_context,
        )
        placeholder.markdown(response)

        # Metadata
        meta_parts = [f"🤖 `{selected_model}`"]
        if rag_used:
            meta_parts.append('<span class="rag-pill">RAG</span>')
        st.markdown(
            '<span style="font-size:0.72rem;color:#6e7681">' + " &nbsp;·&nbsp; ".join(meta_parts) + "</span>",
            unsafe_allow_html=True,
        )

    st.session_state.messages.append({
        "role":    "assistant",
        "content": response,
        "model":   selected_model,
        "rag_used": rag_used,
    })
