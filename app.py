import streamlit as st
import os
import requests
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from datetime import datetime

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Portfolio GPT", page_icon="🤖", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* { font-family: 'Inter', sans-serif; box-sizing: border-box; }

/* Hide Streamlit defaults */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 !important; max-width: 100% !important; }
section[data-testid="stSidebar"] > div { padding-top: 0 !important; }

/* ── SIDEBAR ── */
section[data-testid="stSidebar"] {
    background: #0f0f0f;
    border-right: 1px solid #2a2a2a;
    width: 260px !important;
}
.sidebar-logo {
    padding: 20px 16px 12px;
    border-bottom: 1px solid #2a2a2a;
    margin-bottom: 12px;
}
.sidebar-logo h1 {
    font-size: 18px;
    font-weight: 600;
    color: #ffffff;
    margin: 0;
}
.sidebar-logo p {
    font-size: 11px;
    color: #666;
    margin: 2px 0 0;
}
.new-chat-btn {
    display: flex;
    align-items: center;
    gap: 8px;
    width: calc(100% - 32px);
    margin: 0 16px 16px;
    padding: 10px 14px;
    background: transparent;
    border: 1px solid #3a3a3a;
    border-radius: 8px;
    color: #fff;
    font-size: 13px;
    cursor: pointer;
    transition: background 0.15s;
}
.new-chat-btn:hover { background: #1a1a1a; }

.sidebar-section {
    padding: 0 16px;
    margin-bottom: 16px;
}
.sidebar-label {
    font-size: 10px;
    font-weight: 600;
    color: #555;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 6px;
}
.suggested-q {
    padding: 8px 10px;
    border-radius: 6px;
    font-size: 12px;
    color: #aaa;
    cursor: pointer;
    transition: background 0.15s, color 0.15s;
    margin-bottom: 2px;
    border: none;
    background: transparent;
    text-align: left;
    width: 100%;
}
.suggested-q:hover { background: #1a1a1a; color: #fff; }

.memory-item {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 5px 8px;
    background: #1a1a1a;
    border-radius: 6px;
    font-size: 11px;
    color: #7c7c7c;
    margin-bottom: 4px;
}
.memory-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #10b981;
    flex-shrink: 0;
}

.stat-row {
    display: flex;
    gap: 8px;
    margin-bottom: 8px;
}
.stat-card {
    flex: 1;
    background: #1a1a1a;
    border-radius: 8px;
    padding: 10px;
    text-align: center;
}
.stat-num { font-size: 20px; font-weight: 600; color: #fff; }
.stat-lbl { font-size: 10px; color: #555; margin-top: 2px; }

/* ── MAIN CHAT AREA ── */
.chat-wrapper {
    display: flex;
    flex-direction: column;
    height: 100vh;
    background: #1a1a1a;
}

.chat-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 16px 24px;
    border-bottom: 1px solid #2a2a2a;
    background: #1a1a1a;
    position: sticky;
    top: 0;
    z-index: 10;
}
.chat-header-title {
    font-size: 15px;
    font-weight: 500;
    color: #fff;
}
.model-badge {
    font-size: 11px;
    background: #2a2a2a;
    color: #888;
    padding: 4px 10px;
    border-radius: 20px;
    border: 1px solid #3a3a3a;
}

/* ── WELCOME SCREEN ── */
.welcome-screen {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 40px 24px;
}
.welcome-avatar {
    width: 64px; height: 64px;
    border-radius: 16px;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    display: flex; align-items: center; justify-content: center;
    font-size: 32px;
    margin-bottom: 20px;
    box-shadow: 0 8px 32px rgba(99,102,241,0.3);
}
.welcome-title {
    font-size: 28px;
    font-weight: 600;
    color: #fff;
    margin-bottom: 8px;
    text-align: center;
}
.welcome-sub {
    font-size: 14px;
    color: #666;
    text-align: center;
    margin-bottom: 32px;
    max-width: 400px;
    line-height: 1.6;
}
.suggestion-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
    width: 100%;
    max-width: 560px;
}
.suggestion-card {
    background: #252525;
    border: 1px solid #333;
    border-radius: 12px;
    padding: 14px 16px;
    cursor: pointer;
    transition: all 0.15s;
    text-align: left;
}
.suggestion-card:hover {
    background: #2e2e2e;
    border-color: #6366f1;
    transform: translateY(-1px);
}
.suggestion-icon { font-size: 18px; margin-bottom: 6px; }
.suggestion-text { font-size: 13px; color: #ccc; line-height: 1.4; }

/* ── MESSAGES ── */
.messages-area {
    flex: 1;
    overflow-y: auto;
    padding: 24px;
    max-width: 760px;
    width: 100%;
    margin: 0 auto;
}

.msg-row {
    display: flex;
    gap: 14px;
    margin-bottom: 24px;
    animation: fadeIn 0.2s ease;
}
@keyframes fadeIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }

.msg-avatar {
    width: 32px; height: 32px;
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 15px;
    flex-shrink: 0;
    margin-top: 2px;
}
.msg-avatar.user { background: #6366f1; }
.msg-avatar.bot  { background: linear-gradient(135deg, #6366f1, #8b5cf6); }

.msg-content { flex: 1; }
.msg-name {
    font-size: 12px;
    font-weight: 500;
    color: #666;
    margin-bottom: 4px;
}
.msg-text {
    font-size: 14px;
    color: #e5e5e5;
    line-height: 1.7;
    background: #252525;
    border-radius: 12px;
    padding: 12px 16px;
    border: 1px solid #333;
}
.msg-text.user-bubble {
    background: #6366f120;
    border-color: #6366f140;
    color: #e5e5e5;
}
.source-pill {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    background: #1e1e2e;
    border: 1px solid #6366f140;
    color: #818cf8;
    font-size: 11px;
    padding: 3px 10px;
    border-radius: 20px;
    margin: 8px 4px 0 0;
}

/* ── INPUT BAR ── */
.input-area {
    padding: 16px 24px 24px;
    max-width: 760px;
    width: 100%;
    margin: 0 auto;
}
.input-container {
    background: #252525;
    border: 1px solid #3a3a3a;
    border-radius: 14px;
    padding: 4px 4px 4px 16px;
    display: flex;
    align-items: flex-end;
    gap: 8px;
    transition: border-color 0.15s;
}
.input-container:focus-within { border-color: #6366f1; }

.voice-toggle {
    font-size: 11px;
    color: #666;
    text-align: center;
    margin-top: 10px;
}

/* Streamlit chat input override */
.stChatInput {
    background: transparent !important;
    border: none !important;
}
[data-testid="stChatInput"] {
    background: #252525 !important;
    border: 1px solid #3a3a3a !important;
    border-radius: 14px !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 2px #6366f120 !important;
}
[data-testid="stChatInputTextArea"] {
    color: #e5e5e5 !important;
    background: transparent !important;
}

/* Chat messages */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 4px 0 !important;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
HF_TOKEN   = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")
CHROMA_DIR = "./chroma_db"
DOCS_DIR   = "./docs"

SUGGESTIONS = [
    {"icon": "💻", "text": "What tech stack do you know?"},
    {"icon": "🚀", "text": "Tell me about your best project"},
    {"icon": "💼", "text": "What is your work experience?"},
    {"icon": "🤖", "text": "Do you know machine learning?"},
    {"icon": "📬", "text": "How can I contact you?"},
    {"icon": "🎓", "text": "What is your education background?"},
]

# ── Load vectorstore ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    if os.path.exists(CHROMA_DIR) and len(os.listdir(CHROMA_DIR)) > 0:
        return Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)

    all_docs = []
    for filename in ["about.txt", "skills.txt", "bio.txt"]:
        path = os.path.join(DOCS_DIR, filename)
        if os.path.exists(path):
            loader = TextLoader(path, encoding="utf-8")
            docs = loader.load()
            for d in docs:
                d.metadata["source"] = filename
            all_docs.extend(docs)

    projects_dir = os.path.join(DOCS_DIR, "projects")
    if os.path.exists(projects_dir):
        for fname in os.listdir(projects_dir):
            if fname.endswith((".md", ".txt")):
                loader = TextLoader(os.path.join(projects_dir, fname), encoding="utf-8")
                docs = loader.load()
                for d in docs:
                    d.metadata["source"] = fname
                all_docs.extend(docs)

    if not all_docs:
        st.error("No documents found. Run ingest.py first.")
        st.stop()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(all_docs)
    return Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_DIR)

# ── Session state ─────────────────────────────────────────────────────────────
if "messages"      not in st.session_state: st.session_state.messages      = []
if "turn_count"    not in st.session_state: st.session_state.turn_count    = 0
if "topics_asked"  not in st.session_state: st.session_state.topics_asked  = []
if "visitor_name"  not in st.session_state: st.session_state.visitor_name  = ""
if "voice_enabled" not in st.session_state: st.session_state.voice_enabled = False
if "db_loaded"     not in st.session_state: st.session_state.db_loaded     = False

# ── Memory helpers ────────────────────────────────────────────────────────────
def update_memory(user_msg):
    msg_lower = user_msg.lower()
    for phrase in ["my name is", "i am ", "i'm "]:
        if phrase in msg_lower:
            parts = msg_lower.split(phrase)
            if len(parts) > 1:
                name = parts[1].split()[0].capitalize()
                if len(name) > 1:
                    st.session_state.visitor_name = name
    topics = {
        "skills": ["skill", "stack", "tech", "language", "framework"],
        "projects": ["project", "built", "work", "portfolio"],
        "experience": ["experience", "job", "career", "role"],
        "education": ["study", "degree", "university", "college"],
        "contact": ["contact", "email", "reach", "hire"],
    }
    for topic, keywords in topics.items():
        if any(k in msg_lower for k in keywords):
            if topic not in st.session_state.topics_asked:
                st.session_state.topics_asked.append(topic)

def get_answer(question):
    vs = load_vectorstore()
    docs = vs.similarity_search(question, k=4)
    if not docs:
        return "I don't have information about that.", []

    context = "\n\n".join([
        f"[{os.path.basename(d.metadata.get('source','doc'))}]\n{d.page_content}"
        for d in docs
    ])

    memory_parts = []
    if st.session_state.visitor_name:
        memory_parts.append(f"Visitor name: {st.session_state.visitor_name}")
    if st.session_state.topics_asked:
        memory_parts.append(f"Topics discussed: {', '.join(st.session_state.topics_asked)}")
    if len(st.session_state.messages) >= 4:
        last = st.session_state.messages[-4:]
        history = "\n".join([f"{m['role'].upper()}: {m['content'][:100]}" for m in last])
        memory_parts.append(f"Recent chat:\n{history}")
    memory_ctx = "\n".join(memory_parts)

    if HF_TOKEN:
        try:
            name = st.session_state.visitor_name
            prompt = f"""<s>[INST] You are Portfolio GPT — a professional AI assistant for Harikrishnan's portfolio.
Answer using ONLY the context. Be warm, specific and concise.
{f'Address visitor as {name}.' if name else ''}

{f'Context memory:{chr(10)}{memory_ctx}' if memory_ctx else ''}

Knowledge:
{context}

Question: {question} [/INST]"""

            r = requests.post(
                "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2",
                headers={"Authorization": f"Bearer {HF_TOKEN}"},
                json={"inputs": prompt, "parameters": {"max_new_tokens": 350, "temperature": 0.3, "return_full_text": False}},
                timeout=30,
            )
            result = r.json()
            if isinstance(result, list) and "generated_text" in result[0]:
                return result[0]["generated_text"].strip(), docs
        except Exception:
            pass

    best = docs[0].page_content
    src  = os.path.basename(docs[0].metadata.get("source", "document"))
    return f"Based on **{src}**:\n\n{best}", docs

def speak_text(text):
    clean = text.replace('"', "'").replace('\n', ' ')[:300]
    st.components.v1.html(f"""
    <script>
    const u = new SpeechSynthesisUtterance("{clean}");
    u.rate=0.95; u.pitch=1.0;
    window.speechSynthesis.speak(u);
    </script>""", height=0)

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <h1>🤖 Portfolio GPT</h1>
        <p>AI-powered resume chatbot</p>
    </div>
    """, unsafe_allow_html=True)

    if st.button("✏️  New conversation", use_container_width=True):
        st.session_state.messages     = []
        st.session_state.turn_count   = 0
        st.session_state.topics_asked = []
        st.rerun()

    # Stats
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-label">Session stats</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="stat-row">
        <div class="stat-card">
            <div class="stat-num">{st.session_state.turn_count}</div>
            <div class="stat-lbl">Messages</div>
        </div>
        <div class="stat-card">
            <div class="stat-num">{len(st.session_state.topics_asked)}</div>
            <div class="stat-lbl">Topics</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Memory
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-label">Memory</div>', unsafe_allow_html=True)
    if st.session_state.visitor_name:
        st.markdown(f'<div class="memory-item"><div class="memory-dot"></div>👤 {st.session_state.visitor_name}</div>', unsafe_allow_html=True)
    for t in st.session_state.topics_asked:
        st.markdown(f'<div class="memory-item"><div class="memory-dot"></div>✅ {t}</div>', unsafe_allow_html=True)
    if not st.session_state.visitor_name and not st.session_state.topics_asked:
        st.markdown('<p style="font-size:12px;color:#444;padding:4px 0">No memory yet</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Suggested questions
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-label">Suggested</div>', unsafe_allow_html=True)
    for s in SUGGESTIONS[:4]:
        if st.button(f"{s['icon']} {s['text']}", use_container_width=True, key=f"sq_{s['text']}"):
            st.session_state.messages.append({"role": "user", "content": s["text"]})
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    # Voice toggle
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-label">Settings</div>', unsafe_allow_html=True)
    voice_label = "🔊 Voice ON" if st.session_state.voice_enabled else "🔇 Voice OFF"
    if st.button(voice_label, use_container_width=True):
        st.session_state.voice_enabled = not st.session_state.voice_enabled
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    # Status + footer
    st.markdown("---")
    if HF_TOKEN:
        st.markdown('<p style="font-size:11px;color:#10b981">● Mistral-7B active</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p style="font-size:11px;color:#666">● Running in search mode</p>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:11px;color:#444">Built by Harikrishnan · <a href="https://github.com/harikrishnan200228" style="color:#6366f1">GitHub</a></p>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# MAIN CHAT AREA
# ══════════════════════════════════════════════════════════════════════════════

# Header
st.markdown("""
<div class="chat-header">
    <span class="chat-header-title">🤖 Portfolio GPT</span>
    <span class="model-badge">Mistral-7B · RAG</span>
</div>
""", unsafe_allow_html=True)

# Welcome screen or chat
if not st.session_state.messages:
    st.markdown("""
    <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;padding:60px 24px 20px;">
        <div style="width:72px;height:72px;border-radius:18px;background:linear-gradient(135deg,#6366f1,#8b5cf6);
                    display:flex;align-items:center;justify-content:center;font-size:36px;
                    margin-bottom:20px;box-shadow:0 8px 32px rgba(99,102,241,0.3);">🤖</div>
        <h2 style="font-size:26px;font-weight:600;color:#fff;margin-bottom:8px;text-align:center;">
            How can I help you today?
        </h2>
        <p style="font-size:14px;color:#666;text-align:center;max-width:400px;line-height:1.6;margin-bottom:32px;">
            I'm trained on Harikrishnan's resume and projects.<br>Ask me anything about his experience!
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Suggestion cards in 2 columns
    col1, col2 = st.columns(2)
    for i, s in enumerate(SUGGESTIONS):
        col = col1 if i % 2 == 0 else col2
        with col:
            if st.button(f"{s['icon']} {s['text']}", use_container_width=True, key=f"wc_{s['text']}"):
                st.session_state.messages.append({"role": "user", "content": s["text"]})
                st.rerun()

# Chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🤖"):
        st.markdown(msg["content"], unsafe_allow_html=True)

# Chat input
if prompt := st.chat_input("Message Portfolio GPT..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner(""):
            try:
                answer, source_docs = get_answer(prompt)

                seen, pills = set(), ""
                for doc in source_docs:
                    src = os.path.basename(doc.metadata.get("source", "document"))
                    if src not in seen:
                        seen.add(src)
                        pills += f'<span class="source-pill">📄 {src}</span>'

                full = answer
                if pills:
                    full += f"<br><div style='margin-top:8px'>{pills}</div>"

                st.markdown(full, unsafe_allow_html=True)

                if st.session_state.voice_enabled:
                    speak_text(answer)

                update_memory(prompt)
                st.session_state.turn_count += 1
                st.session_state.messages.append({"role": "assistant", "content": full})

            except Exception as e:
                st.error(f"Error: {e}")