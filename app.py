import streamlit as st
import os
import requests
import json
import base64
from datetime import datetime
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Portfolio GPT", page_icon="🤖", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.main { max-width: 780px; }

.hero {
    background: linear-gradient(135deg, #667eea33, #764ba233);
    border: 1px solid #a78bfa55;
    border-radius: 16px;
    padding: 1.4rem 1.8rem;
    margin-bottom: 1.5rem;
}
.hero h2 { margin: 0 0 4px; font-size: 20px; font-weight: 600; }
.hero p  { margin: 0; font-size: 14px; color: #6b7280; }

.source-badge {
    display: inline-block;
    background: #EEEDFE;
    color: #3C3489;
    font-size: 11px;
    font-weight: 500;
    padding: 2px 10px;
    border-radius: 20px;
    margin-right: 5px;
}
.memory-chip {
    display: inline-block;
    background: #E1F5EE;
    color: #065f46;
    font-size: 11px;
    padding: 2px 8px;
    border-radius: 20px;
    margin-right: 4px;
    margin-bottom: 4px;
}
.stat-box {
    background: var(--background-color);
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    padding: 10px 14px;
    text-align: center;
    margin-bottom: 8px;
}
.stat-num { font-size: 22px; font-weight: 600; }
.stat-label { font-size: 11px; color: #9ca3af; }

.voice-btn {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
    border: none;
    border-radius: 50%;
    width: 42px; height: 42px;
    font-size: 18px;
    cursor: pointer;
    display: flex; align-items: center; justify-content: center;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
HF_TOKEN   = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")
CHROMA_DIR = "./chroma_db"
DOCS_DIR   = "./docs"

SAMPLE_QUESTIONS = [
    "What tech stack do you know?",
    "Tell me about your best project",
    "What is your work experience?",
    "Do you know machine learning?",
    "What are your soft skills?",
    "How can I contact you?",
]

# ── Load vectorstore ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="🔍 Loading knowledge base...")
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    if os.path.exists(CHROMA_DIR) and len(os.listdir(CHROMA_DIR)) > 0:
        return Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)

    # Auto-build from docs/
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
        st.error("No documents found in docs/ — run ingest.py first.")
        st.stop()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(all_docs)
    return Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_DIR)

vectorstore = load_vectorstore()

# ── Session state init ────────────────────────────────────────────────────────
if "messages"      not in st.session_state: st.session_state.messages      = []
if "memory"        not in st.session_state: st.session_state.memory        = {}   # key facts remembered
if "turn_count"    not in st.session_state: st.session_state.turn_count    = 0
if "topics_asked"  not in st.session_state: st.session_state.topics_asked  = []
if "visitor_name"  not in st.session_state: st.session_state.visitor_name  = ""
if "voice_enabled" not in st.session_state: st.session_state.voice_enabled = False

# ── Memory helpers ────────────────────────────────────────────────────────────
def update_memory(user_msg: str, assistant_msg: str):
    """Extract and store key facts from the conversation."""
    msg_lower = user_msg.lower()
    # Remember visitor name
    for phrase in ["my name is", "i am ", "i'm "]:
        if phrase in msg_lower:
            parts = msg_lower.split(phrase)
            if len(parts) > 1:
                name = parts[1].split()[0].capitalize()
                if len(name) > 1:
                    st.session_state.memory["visitor_name"] = name
                    st.session_state.visitor_name = name
    # Track topics
    topics = {
        "skills": ["skill", "stack", "technology", "language", "framework"],
        "projects": ["project", "built", "work", "portfolio"],
        "experience": ["experience", "job", "career", "role"],
        "education": ["study", "degree", "university", "college"],
        "contact": ["contact", "email", "reach", "hire"],
    }
    for topic, keywords in topics.items():
        if any(k in msg_lower for k in keywords):
            if topic not in st.session_state.topics_asked:
                st.session_state.topics_asked.append(topic)

def build_memory_context() -> str:
    """Build a memory string to inject into the prompt."""
    parts = []
    if st.session_state.visitor_name:
        parts.append(f"Visitor's name: {st.session_state.visitor_name}")
    if st.session_state.turn_count > 0:
        parts.append(f"This is turn {st.session_state.turn_count + 1} of the conversation.")
    if st.session_state.topics_asked:
        parts.append(f"Topics already discussed: {', '.join(st.session_state.topics_asked)}")
    # Last 2 exchanges for context
    if len(st.session_state.messages) >= 4:
        last = st.session_state.messages[-4:]
        history = "\n".join([f"{m['role'].upper()}: {m['content'][:120]}" for m in last])
        parts.append(f"Recent conversation:\n{history}")
    return "\n".join(parts) if parts else ""

# ── Answer engine ─────────────────────────────────────────────────────────────
def get_answer(question: str):
    docs = vectorstore.similarity_search(question, k=4)
    if not docs:
        return "I don't have information about that in my knowledge base.", []

    context = "\n\n".join([
        f"[{os.path.basename(d.metadata.get('source', 'doc'))}]\n{d.page_content}"
        for d in docs
    ])
    memory_ctx = build_memory_context()
    visitor    = st.session_state.visitor_name

    if HF_TOKEN:
        try:
            greeting = f"Hi {visitor}! " if visitor else ""
            prompt = f"""<s>[INST] You are Portfolio GPT — an AI assistant for Harik's professional portfolio.
Answer using ONLY the context provided. Be warm, specific, and professional.
{f"Address the visitor as {visitor}." if visitor else ""}
If the answer isn't in context, say "I don't have that information, but feel free to reach out directly."

{f"Memory & Conversation Context:{chr(10)}{memory_ctx}" if memory_ctx else ""}

Knowledge Base Context:
{context}

Question: {question} [/INST]
{greeting}"""

            response = requests.post(
                "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2",
                headers={"Authorization": f"Bearer {HF_TOKEN}"},
                json={
                    "inputs": prompt,
                    "parameters": {
                        "max_new_tokens": 350,
                        "temperature": 0.3,
                        "return_full_text": False,
                    }
                },
                timeout=30,
            )
            result = response.json()
            if isinstance(result, list) and "generated_text" in result[0]:
                return result[0]["generated_text"].strip(), docs
        except Exception:
            pass

    # Fallback — best chunk
    best   = docs[0].page_content
    source = os.path.basename(docs[0].metadata.get("source", "document"))
    prefix = f"Hi {visitor}! " if visitor else ""
    return f"{prefix}Based on **{source}**:\n\n{best}", docs

# ── TTS helper (browser-based, free) ─────────────────────────────────────────
def speak_text(text: str):
    """Inject browser Web Speech API to read text aloud."""
    clean = text.replace('"', "'").replace('\n', ' ')[:300]
    js = f"""
    <script>
    const msg = new SpeechSynthesisUtterance("{clean}");
    msg.rate = 0.95;
    msg.pitch = 1.0;
    window.speechSynthesis.speak(msg);
    </script>
    """
    st.components.v1.html(js, height=0)

# ═══════════════════════════════════════════════════════════════════════════════
# UI LAYOUT
# ═══════════════════════════════════════════════════════════════════════════════

# ── Hero header ───────────────────────────────────────────────────────────────
greeting_name = f", {st.session_state.visitor_name}" if st.session_state.visitor_name else ""
st.markdown(f"""
<div class="hero">
  <h2>🤖 Portfolio GPT</h2>
  <p>Hi{greeting_name}! Ask me anything about Harikrishnan's skills, projects, and experience — I'll answer with cited sources.</p>
</div>
""", unsafe_allow_html=True)

# ── Top controls row ──────────────────────────────────────────────────────────
col1, col2, col3 = st.columns([2, 1, 1])
with col2:
    voice_label = "🔊 Voice ON" if st.session_state.voice_enabled else "🔇 Voice OFF"
    if st.button(voice_label, use_container_width=True):
        st.session_state.voice_enabled = not st.session_state.voice_enabled
        st.rerun()
with col3:
    if st.button("🗑️ Clear", use_container_width=True):
        st.session_state.messages     = []
        st.session_state.turn_count   = 0
        st.session_state.topics_asked = []
        st.rerun()

# ── Sample question buttons ───────────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown("**✨ Try asking:**")
    cols = st.columns(2)
    for i, q in enumerate(SAMPLE_QUESTIONS):
        if cols[i % 2].button(q, use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": q})
            st.rerun()

# ── Chat history ──────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)

# ── Chat input ────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask anything about my experience..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                answer, source_docs = get_answer(prompt)

                # Source badges
                seen, badges = set(), ""
                for doc in source_docs:
                    src = os.path.basename(doc.metadata.get("source", "document"))
                    if src not in seen:
                        seen.add(src)
                        badges += f'<span class="source-badge">📄 {src}</span>'

                full_response = answer
                if badges:
                    full_response += f"<br><br><small>Sources: {badges}</small>"

                st.markdown(full_response, unsafe_allow_html=True)

                # Voice output
                if st.session_state.voice_enabled:
                    speak_text(answer)

                # Update memory
                update_memory(prompt, answer)
                st.session_state.turn_count += 1

                st.session_state.messages.append(
                    {"role": "assistant", "content": full_response}
                )

            except Exception as e:
                st.error(f"Error: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🤖 Portfolio GPT")
    st.caption("Powered by RAG + Memory + Voice")
    st.divider()

    # Stats
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f'<div class="stat-box"><div class="stat-num">{st.session_state.turn_count}</div><div class="stat-label">Turns</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="stat-box"><div class="stat-num">{len(st.session_state.topics_asked)}</div><div class="stat-label">Topics</div></div>', unsafe_allow_html=True)

    st.divider()

    # Memory panel
    st.markdown("#### 🧠 Memory")
    if st.session_state.visitor_name:
        st.markdown(f'<span class="memory-chip">👤 {st.session_state.visitor_name}</span>', unsafe_allow_html=True)
    if st.session_state.topics_asked:
        for t in st.session_state.topics_asked:
            st.markdown(f'<span class="memory-chip">✅ {t}</span>', unsafe_allow_html=True)
    else:
        st.caption("No memory yet — start chatting!")

    st.divider()

    # Features
    st.markdown("#### ⚡ Features")
    st.markdown("🧠 **Conversation memory**\n\nRemembers your name and past topics")
    st.markdown("🔊 **Voice output**\n\nToggle voice to hear answers")
    st.markdown("📄 **Cited answers**\n\nSources shown for every answer")
    st.markdown("🤖 **Mistral-7B LLM**\n\nFree HuggingFace inference")

    st.divider()

    # Status
    if HF_TOKEN:
        st.success("✅ Mistral-7B active")
    else:
        st.info("ℹ️ Add HF token for AI answers")
        st.code("HUGGINGFACEHUB_API_TOKEN=hf_...", language="bash")

    st.divider()
    st.markdown("**Tech Stack**")
    st.markdown("- `all-MiniLM-L6-v2` embeddings\n- ChromaDB vector DB\n- Mistral-7B LLM\n- Streamlit UI\n- Browser Web Speech API")
    st.divider()
    st.caption("Built by Harikrishnan · [GitHub](https://github.com/harikrishnan2002)")
