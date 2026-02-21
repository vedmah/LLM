import streamlit as st
import os
import time
from datetime import datetime

# ─── SDK Imports (graceful if missing) ───────────────────────────────────────
try:
    from openai import OpenAI
    OPENAI_OK = True
except ImportError:
    OPENAI_OK = False

try:
    import google.generativeai as genai
    GEMINI_OK = True
except ImportError:
    GEMINI_OK = False

try:
    from groq import Groq
    GROQ_OK = True
except ImportError:
    GROQ_OK = False

try:
    import anthropic
    CLAUDE_OK = True
except ImportError:
    CLAUDE_OK = False

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Multi-LLM Chat Hub",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# MODEL REGISTRY
# ─────────────────────────────────────────────────────────────────────────────
MODELS = {
    "OpenAI GPT-4o": {
        "provider":    "openai",
        "model_id":    "gpt-4o",
        "icon":        "🟢",
        "color":       "#10a37f",
        "bg":          "#f0fdf9",
        "border":      "#6ee7d4",
        "sdk_ok":      OPENAI_OK,
        "env_key":     "OPENAI_API_KEY",
        "description": "GPT-4o — OpenAI's flagship multimodal model",
    },
    "Google Gemini": {
        "provider":    "gemini",
        "model_id":    "gemini-1.5-flash",
        "icon":        "🔵",
        "color":       "#4285f4",
        "bg":          "#eff6ff",
        "border":      "#93c5fd",
        "sdk_ok":      GEMINI_OK,
        "env_key":     "GEMINI_API_KEY",
        "description": "Gemini 1.5 Flash — Google's fast multimodal model",
    },
    "Groq (LLaMA 3)": {
        "provider":    "groq",
        "model_id":    "llama3-8b-8192",
        "icon":        "🟡",
        "color":       "#f59e0b",
        "bg":          "#fffbeb",
        "border":      "#fcd34d",
        "sdk_ok":      GROQ_OK,
        "env_key":     "GROQ_API_KEY",
        "description": "LLaMA 3 8B on Groq — ultra-fast inference",
    },
    "Claude Sonnet": {
        "provider":    "claude",
        "model_id":    "claude-3-5-sonnet-20241022",
        "icon":        "🟠",
        "color":       "#d97706",
        "bg":          "#fffbeb",
        "border":      "#fed7aa",
        "sdk_ok":      CLAUDE_OK,
        "env_key":     "ANTHROPIC_API_KEY",
        "description": "Claude 3.5 Sonnet — Anthropic's balanced model",
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
    --bg:       #f3f4f6;
    --surface:  #ffffff;
    --border:   #e5e7eb;
    --text:     #111827;
    --text-sec: #6b7280;
    --accent:   #4f46e5;
    --accent-lt:#eef2ff;
}
html, body, [class*="css"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Inter', sans-serif !important;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.2rem 1.8rem 2rem !important; max-width: 1300px; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] > div {
    background: #1e1b4b !important;
    border-right: none !important;
}
section[data-testid="stSidebar"] * { color: #e0e7ff !important; }
section[data-testid="stSidebar"] .stSelectbox > div > div {
    background: #312e81 !important;
    border: 1px solid #4338ca !important;
    border-radius: 8px !important;
}
section[data-testid="stSidebar"] .stTextInput input {
    background: #312e81 !important;
    border: 1px solid #4338ca !important;
    border-radius: 8px !important;
    color: #e0e7ff !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 12px !important;
}
section[data-testid="stSidebar"] hr { border-color: #3730a3 !important; }

/* ── Model selector cards ── */
.model-card {
    border: 2px solid var(--border);
    border-radius: 12px;
    padding: 14px 18px;
    margin-bottom: 8px;
    cursor: pointer;
    display: flex;
    align-items: center;
    gap: 12px;
    background: var(--surface);
    transition: all 0.15s;
}
.model-card.active {
    border-color: var(--accent);
    background: var(--accent-lt);
    box-shadow: 0 2px 12px rgba(79,70,229,0.12);
}
.model-icon { font-size: 22px; }
.model-name { font-size: 14px; font-weight: 700; }
.model-desc { font-size: 11px; color: var(--text-sec); margin-top: 2px; }

/* ── Chat bubbles ── */
.chat-wrap { display: flex; flex-direction: column; gap: 14px; padding: 4px 0; }

.bubble-user {
    display: flex;
    justify-content: flex-end;
}
.bubble-user .bubble-inner {
    background: #4f46e5;
    color: #fff;
    border-radius: 18px 18px 4px 18px;
    padding: 12px 18px;
    max-width: 72%;
    font-size: 14px;
    line-height: 1.6;
    box-shadow: 0 2px 8px rgba(79,70,229,0.18);
}

.bubble-ai {
    display: flex;
    justify-content: flex-start;
    gap: 10px;
    align-items: flex-start;
}
.bubble-avatar {
    width: 32px; height: 32px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 16px;
    flex-shrink: 0;
    margin-top: 2px;
}
.bubble-ai .bubble-inner {
    border-radius: 4px 18px 18px 18px;
    padding: 12px 18px;
    max-width: 80%;
    font-size: 14px;
    line-height: 1.7;
    box-shadow: 0 1px 4px rgba(0,0,0,0.07);
}

.bubble-meta {
    font-size: 10px;
    color: var(--text-sec);
    margin-top: 4px;
    padding: 0 6px;
}

/* ── Input area ── */
.chat-input-wrap {
    position: sticky;
    bottom: 0;
    background: var(--bg);
    padding: 12px 0 4px;
    border-top: 1px solid var(--border);
    margin-top: 10px;
}

/* ── Provider badge ── */
.provider-badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    border-radius: 20px;
    padding: 3px 10px;
    font-size: 11px;
    font-weight: 600;
    margin-bottom: 4px;
}

/* ── Header bar ── */
.hub-header {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
    border-radius: 14px;
    padding: 18px 24px;
    color: white;
    margin-bottom: 1rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: 0 4px 20px rgba(79,70,229,0.22);
}
.hub-title { font-size: 22px; font-weight: 700; letter-spacing: -0.02em; }
.hub-sub   { font-size: 12px; opacity: 0.75; margin-top: 3px; }
.hub-stats { text-align: right; }
.hub-stats-num { font-family: 'IBM Plex Mono', monospace; font-size: 22px; font-weight: 600; }
.hub-stats-lbl { font-size: 10px; opacity: 0.7; text-transform: uppercase; letter-spacing: 0.06em; }

/* ── Empty state ── */
.empty-state {
    text-align: center;
    padding: 60px 20px;
    color: var(--text-sec);
}
.empty-icon { font-size: 48px; margin-bottom: 12px; }
.empty-title { font-size: 16px; font-weight: 600; color: var(--text); margin-bottom: 6px; }
.empty-sub   { font-size: 13px; }

/* ── Code blocks ── */
pre {
    background: #1e1b4b !important;
    color: #e0e7ff !important;
    border-radius: 8px !important;
    padding: 14px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 12px !important;
    overflow-x: auto;
}

/* ── Streamlit tweaks ── */
.stButton > button {
    background: var(--accent) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 13px !important;
}
.stButton > button:hover { background: #4338ca !important; }
.stTextArea textarea {
    background: var(--surface) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 10px !important;
    font-size: 14px !important;
    font-family: 'Inter', sans-serif !important;
    resize: none !important;
}
.stTextArea textarea:focus { border-color: var(--accent) !important; }
.stSlider, .stCheckbox { color: #e0e7ff !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "OpenAI GPT-4o"

# Per-model conversation history
for model_name in MODELS:
    key = f"history_{model_name}"
    if key not in st.session_state:
        st.session_state[key] = []

if "total_messages" not in st.session_state:
    st.session_state.total_messages = 0

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding: 12px 4px 20px;'>
        <div style='font-size:20px;font-weight:700;color:#e0e7ff;letter-spacing:-0.02em;'>🤖 LLM Hub</div>
        <div style='font-size:11px;color:#a5b4fc;margin-top:3px;'>Multi-Model Chat Interface</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<div style='font-size:10px;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:#818cf8;margin-bottom:8px;'>Choose Model</div>", unsafe_allow_html=True)

    for model_name, meta in MODELS.items():
        is_active = st.session_state.selected_model == model_name
        sdk_warn  = "" if meta["sdk_ok"] else " ⚠️"
        border_c  = "#818cf8" if is_active else "#3730a3"
        bg_c      = "#312e81" if is_active else "transparent"
        if st.button(
            f"{meta['icon']}  {model_name}{sdk_warn}",
            key=f"sel_{model_name}",
            use_container_width=True,
        ):
            st.session_state.selected_model = model_name
            st.rerun()

    st.markdown("---")

    # API Key inputs
    st.markdown("<div style='font-size:10px;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:#818cf8;margin-bottom:8px;'>API Keys</div>", unsafe_allow_html=True)

    current_model_meta = MODELS[st.session_state.selected_model]
    env_key = current_model_meta["env_key"]

    # Show key input for active model
    saved_key = st.session_state.get(f"apikey_{env_key}", os.environ.get(env_key, ""))
    new_key = st.text_input(
        f"{env_key}",
        value=saved_key,
        type="password",
        key=f"input_{env_key}",
        placeholder="sk-... / AIza... / gsk_...",
    )
    if new_key:
        st.session_state[f"apikey_{env_key}"] = new_key

    st.markdown("---")

    # Model settings
    st.markdown("<div style='font-size:10px;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:#818cf8;margin-bottom:8px;'>Settings</div>", unsafe_allow_html=True)
    max_tokens = st.slider("Max Tokens", 256, 4096, 1024, 128)
    temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.05)
    system_prompt = st.text_area(
        "System Prompt",
        value="You are a helpful, knowledgeable AI assistant. Answer clearly and concisely.",
        height=90,
    )

    st.markdown("---")

    # Clear history
    active_history_key = f"history_{st.session_state.selected_model}"
    hist_len = len(st.session_state[active_history_key])
    st.markdown(f"<div style='font-size:11px;color:#a5b4fc;margin-bottom:6px;'>Current session: {hist_len} messages</div>", unsafe_allow_html=True)

    col_clr1, col_clr2 = st.columns(2)
    with col_clr1:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state[active_history_key] = []
            st.rerun()
    with col_clr2:
        if st.button("Clear All", use_container_width=True):
            for m in MODELS:
                st.session_state[f"history_{m}"] = []
            st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# API CALL ROUTING  —  match / case
# ─────────────────────────────────────────────────────────────────────────────
def get_api_key(env_key: str) -> str:
    """Retrieve API key from session state or environment."""
    return st.session_state.get(f"apikey_{env_key}", os.environ.get(env_key, ""))


def call_model(model_name: str, messages: list, max_tok: int, temp: float, sys_prompt: str) -> str:
    """
    Route the API call to the correct provider using match/case.
    messages: list of {"role": "user"|"assistant", "content": str}
    Returns the assistant reply string.
    """
    meta = MODELS[model_name]
    api_key = get_api_key(meta["env_key"])

    if not api_key:
        return f"⚠️  No API key set for **{model_name}**. Please enter your `{meta['env_key']}` in the sidebar."

    provider = meta["provider"]

    match provider:

        # ── OpenAI ────────────────────────────────────────────────────────
        case "openai":
            if not OPENAI_OK:
                return "❌  `openai` package not installed. Run: `pip install openai`"
            client = OpenAI(api_key=api_key)
            formatted = [{"role": "system", "content": sys_prompt}]
            for m in messages:
                formatted.append({"role": m["role"], "content": m["content"]})
            response = client.chat.completions.create(
                model=meta["model_id"],
                messages=formatted,
                max_tokens=max_tok,
                temperature=temp,
            )
            return response.choices[0].message.content

        # ── Google Gemini ─────────────────────────────────────────────────
        case "gemini":
            if not GEMINI_OK:
                return "❌  `google-generativeai` package not installed. Run: `pip install google-generativeai`"
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(
                model_name=meta["model_id"],
                system_instruction=sys_prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=max_tok,
                    temperature=temp,
                ),
            )
            # Build Gemini history format
            gemini_history = []
            for m in messages[:-1]:   # all except last user message
                gemini_history.append({
                    "role": "user" if m["role"] == "user" else "model",
                    "parts": [m["content"]],
                })
            chat = model.start_chat(history=gemini_history)
            response = chat.send_message(messages[-1]["content"])
            return response.text

        # ── Groq ──────────────────────────────────────────────────────────
        case "groq":
            if not GROQ_OK:
                return "❌  `groq` package not installed. Run: `pip install groq`"
            client = Groq(api_key=api_key)
            formatted = [{"role": "system", "content": sys_prompt}]
            for m in messages:
                formatted.append({"role": m["role"], "content": m["content"]})
            response = client.chat.completions.create(
                model=meta["model_id"],
                messages=formatted,
                max_tokens=max_tok,
                temperature=temp,
            )
            return response.choices[0].message.content

        # ── Anthropic Claude ──────────────────────────────────────────────
        case "claude":
            if not CLAUDE_OK:
                return "❌  `anthropic` package not installed. Run: `pip install anthropic`"
            client = anthropic.Anthropic(api_key=api_key)
            formatted = []
            for m in messages:
                formatted.append({"role": m["role"], "content": m["content"]})
            response = client.messages.create(
                model=meta["model_id"],
                max_tokens=max_tok,
                system=sys_prompt,
                messages=formatted,
            )
            return response.content[0].text

        # ── Unknown provider ──────────────────────────────────────────────
        case _:
            return f"❌  Unknown provider: `{provider}`"


# ─────────────────────────────────────────────────────────────────────────────
# MAIN UI
# ─────────────────────────────────────────────────────────────────────────────
selected      = st.session_state.selected_model
meta          = MODELS[selected]
history_key   = f"history_{selected}"
chat_history  = st.session_state[history_key]
total_msgs    = sum(len(st.session_state[f"history_{m}"]) for m in MODELS)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hub-header">
    <div>
        <div class="hub-title">{meta['icon']}  {selected}</div>
        <div class="hub-sub">{meta['description']}</div>
    </div>
    <div class="hub-stats">
        <div class="hub-stats-num">{len(chat_history) // 2}</div>
        <div class="hub-stats-lbl">Exchanges this session</div>
    </div>
</div>""", unsafe_allow_html=True)

# ── Model quick-switch tabs ───────────────────────────────────────────────────
tab_labels = [f"{m['icon']} {name.split()[0]}" for name, m in MODELS.items()]
model_names = list(MODELS.keys())
tabs = st.tabs(tab_labels)

for i, (tab, model_name) in enumerate(zip(tabs, model_names)):
    with tab:
        if model_name != selected:
            h = st.session_state[f"history_{model_name}"]
            m = MODELS[model_name]
            if h:
                st.markdown(f"<div style='color:#9ca3af;font-size:12px;padding:6px 0;'>{len(h)//2} exchange(s) in history. Click the sidebar to switch.</div>", unsafe_allow_html=True)
                # Preview last exchange
                for msg in h[-2:]:
                    role_label = "You" if msg["role"] == "user" else model_name
                    st.markdown(f"**{role_label}:** {msg['content'][:200]}{'…' if len(msg['content'])>200 else ''}")
            else:
                st.markdown(f"<div style='color:#9ca3af;font-size:12px;'>No history yet for {model_name}.</div>", unsafe_allow_html=True)
        else:
            # ── Active model chat view ────────────────────────────────────
            chat_container = st.container()
            with chat_container:
                if not chat_history:
                    st.markdown(f"""
                    <div class="empty-state">
                        <div class="empty-icon">{meta['icon']}</div>
                        <div class="empty-title">Start chatting with {selected}</div>
                        <div class="empty-sub">{meta['description']}<br><br>
                        Your conversation history is preserved per model.</div>
                    </div>""", unsafe_allow_html=True)
                else:
                    for msg in chat_history:
                        if msg["role"] == "user":
                            st.markdown(f"""
                            <div class="bubble-user">
                                <div>
                                    <div class="bubble-inner">{msg['content']}</div>
                                    <div class="bubble-meta" style="text-align:right;">You · {msg.get('time','')}</div>
                                </div>
                            </div>""", unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="bubble-ai">
                                <div class="bubble-avatar" style="background:{meta['bg']};border:2px solid {meta['border']};">{meta['icon']}</div>
                                <div>
                                    <div class="bubble-inner" style="background:{meta['bg']};border:1px solid {meta['border']};">
                                        {msg['content']}
                                    </div>
                                    <div class="bubble-meta">{selected} · {msg.get('time','')}</div>
                                </div>
                            </div>""", unsafe_allow_html=True)

            # ── Input row ─────────────────────────────────────────────────
            st.markdown('<div class="chat-input-wrap">', unsafe_allow_html=True)
            input_col, btn_col = st.columns([10, 1])
            with input_col:
                user_input = st.text_area(
                    "Message",
                    placeholder=f"Message {selected}… (Shift+Enter for new line)",
                    label_visibility="collapsed",
                    key=f"input_{model_name}",
                    height=68,
                )
            with btn_col:
                st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
                send = st.button("Send ➤", key=f"send_{model_name}", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # ── Send logic ────────────────────────────────────────────────
            if send and user_input.strip():
                now = datetime.now().strftime("%H:%M")

                # Append user message to this model's history
                st.session_state[history_key].append({
                    "role": "user",
                    "content": user_input.strip(),
                    "time": now,
                })

                # Build messages for API (role + content only)
                api_messages = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state[history_key]
                ]

                # Call the model
                with st.spinner(f"{meta['icon']}  {selected} is thinking…"):
                    try:
                        reply = call_model(
                            model_name=selected,
                            messages=api_messages,
                            max_tok=max_tokens,
                            temp=temperature,
                            sys_prompt=system_prompt,
                        )
                    except Exception as e:
                        reply = f"❌  Error: `{type(e).__name__}: {e}`"

                # Append assistant reply
                st.session_state[history_key].append({
                    "role": "assistant",
                    "content": reply,
                    "time": datetime.now().strftime("%H:%M"),
                })
                st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style='text-align:center;color:#9ca3af;font-size:10px;margin-top:1.5rem;
     padding-top:1rem;border-top:1px solid #e5e7eb;'>
    Multi-LLM Chat Hub &nbsp;·&nbsp; OpenAI · Gemini · Groq · Claude &nbsp;·&nbsp;
    {datetime.now().strftime('%d %b %Y')} &nbsp;·&nbsp;
    History: {total_msgs} total messages across all models
</div>""", unsafe_allow_html=True)
