import os
import streamlit as st
import pypdf
import chromadb
from openai import OpenAI

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "Chat with PDF",
    page_icon  = "📄",
    layout     = "wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Background */
.stApp {
    background: #0f0f0f;
    color: #e8e3db;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #161616;
    border-right: 1px solid #2a2a2a;
}

/* Title */
.main-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.4rem;
    color: #e8e3db;
    letter-spacing: -0.02em;
    margin-bottom: 0;
    line-height: 1.1;
}
.main-subtitle {
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    color: #c8a96e;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-top: 4px;
    margin-bottom: 2rem;
}

/* Chat messages */
.chat-user {
    background: #1e1e1e;
    border: 1px solid #2a2a2a;
    border-radius: 12px 12px 4px 12px;
    padding: 14px 18px;
    margin: 8px 0;
    margin-left: 15%;
    font-size: 0.95rem;
    color: #e8e3db;
}
.chat-assistant {
    background: #1a1a1a;
    border: 1px solid #c8a96e33;
    border-left: 3px solid #c8a96e;
    border-radius: 4px 12px 12px 12px;
    padding: 14px 18px;
    margin: 8px 0;
    margin-right: 15%;
    font-size: 0.95rem;
    color: #e8e3db;
    line-height: 1.7;
}
.chat-sources {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: #c8a96e;
    margin-top: 8px;
    padding-top: 8px;
    border-top: 1px solid #2a2a2a;
}

/* Input box */
.stTextInput > div > div > input {
    background: #1e1e1e !important;
    border: 1px solid #2a2a2a !important;
    border-radius: 8px !important;
    color: #e8e3db !important;
    font-family: 'DM Sans', sans-serif !important;
    padding: 12px 16px !important;
}
.stTextInput > div > div > input:focus {
    border-color: #c8a96e !important;
    box-shadow: 0 0 0 2px #c8a96e22 !important;
}

/* Buttons */
.stButton > button {
    background: #c8a96e !important;
    color: #0f0f0f !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    padding: 10px 24px !important;
    transition: all 0.2s ease !important;
    width: 100%;
}
.stButton > button:hover {
    background: #d4b97e !important;
    transform: translateY(-1px);
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: #1a1a1a;
    border: 1px dashed #2a2a2a;
    border-radius: 10px;
    padding: 8px;
}

/* Status badges */
.status-ready {
    background: #1a2e1a;
    border: 1px solid #2d5a2d;
    color: #6fcf6f;
    border-radius: 6px;
    padding: 8px 14px;
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    text-align: center;
}
.status-waiting {
    background: #2a1f0f;
    border: 1px solid #5a3d1a;
    color: #c8a96e;
    border-radius: 6px;
    padding: 8px 14px;
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    text-align: center;
}

/* Divider */
hr { border-color: #2a2a2a !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0f0f0f; }
::-webkit-scrollbar-thumb { background: #2a2a2a; border-radius: 2px; }

/* Hide streamlit branding */
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Config ────────────────────────────────────────────────────────────────────
EMBED_MODEL   = "text-embedding-3-small"
LLM_MODEL     = "gpt-4o"
CHUNK_SIZE    = 1000
CHUNK_OVERLAP = 200
TOP_K         = 4


# ── Session state ─────────────────────────────────────────────────────────────
if "chat_history"   not in st.session_state: st.session_state.chat_history   = []
if "collection"     not in st.session_state: st.session_state.collection     = None
if "pdf_ready"      not in st.session_state: st.session_state.pdf_ready      = False
if "pdf_name"       not in st.session_state: st.session_state.pdf_name       = ""
if "openai_client"  not in st.session_state: st.session_state.openai_client  = None
if "messages"       not in st.session_state: st.session_state.messages       = []


# ── Helpers ───────────────────────────────────────────────────────────────────
def split_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += chunk_size - overlap
    return chunks


def process_pdf(uploaded_file, client):
    reader     = pypdf.PdfReader(uploaded_file)
    all_chunks = []
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            for chunk in split_text(text):
                all_chunks.append({"text": chunk, "page": page_num + 1})

    # Build ChromaDB collection (in-memory)
    db = chromadb.Client()
    try:
        db.delete_collection("pdf_rag")
    except:
        pass
    collection = db.create_collection("pdf_rag")

    # Embed in batches
    BATCH = 50
    progress = st.progress(0, text="Creating embeddings…")
    for i in range(0, len(all_chunks), BATCH):
        batch      = all_chunks[i:i + BATCH]
        texts      = [c["text"] for c in batch]
        response   = client.embeddings.create(model=EMBED_MODEL, input=texts)
        embeddings = [r.embedding for r in response.data]
        collection.add(
            ids        = [str(i + j) for j in range(len(batch))],
            documents  = texts,
            embeddings = embeddings,
            metadatas  = [{"page": c["page"]} for c in batch],
        )
        progress.progress(min((i + BATCH) / len(all_chunks), 1.0),
                          text=f"Embedding chunks {i}–{i+len(batch)} of {len(all_chunks)}…")
    progress.empty()
    return collection, len(all_chunks), len(reader.pages)


def ask(question, collection, client, chat_history):
    # Embed question
    q_emb = client.embeddings.create(model=EMBED_MODEL, input=[question]).data[0].embedding

    # Retrieve relevant chunks
    results      = collection.query(query_embeddings=[q_emb], n_results=TOP_K)
    context      = "\n\n".join(results["documents"][0])
    source_pages = sorted({m["page"] for m in results["metadatas"][0]})

    # Build messages
    system_msg = {
        "role"   : "system",
        "content": f"""You are a helpful assistant that answers questions based on the
PDF document context below. Use ONLY this context to answer.
If the answer is not in the context, say "I couldn't find that in the document."

Context:
{context}"""
    }
    messages = [system_msg] + chat_history + [{"role": "user", "content": question}]

    # Call GPT
    response = client.chat.completions.create(
        model       = LLM_MODEL,
        messages    = messages,
        max_tokens  = 1024,
        temperature = 0.2,
    )
    answer = response.choices[0].message.content

    # Update history
    chat_history.append({"role": "user",      "content": question})
    chat_history.append({"role": "assistant", "content": answer})

    return answer, source_pages


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="main-title">Chat<br>with PDF</p>', unsafe_allow_html=True)
    st.markdown('<p class="main-subtitle">RAG · GPT-4o · ChromaDB</p>', unsafe_allow_html=True)
    st.markdown("---")

    # API Key
    st.markdown("**OpenAI API Key**")
    api_key = st.text_input("", type="password", placeholder="sk-...", label_visibility="collapsed")
    if api_key:
        st.session_state.openai_client = OpenAI(api_key=api_key)

    st.markdown("---")

    # PDF Upload
    st.markdown("**Upload PDF**")
    uploaded_file = st.file_uploader("", type="pdf", label_visibility="collapsed")

    if uploaded_file and st.session_state.openai_client:
        if st.button("⚡ Process PDF"):
            with st.spinner("Reading and embedding PDF…"):
                try:
                    collection, n_chunks, n_pages = process_pdf(
                        uploaded_file, st.session_state.openai_client
                    )
                    st.session_state.collection    = collection
                    st.session_state.pdf_ready     = True
                    st.session_state.pdf_name      = uploaded_file.name
                    st.session_state.chat_history  = []
                    st.session_state.messages      = []
                    st.success(f"✅ {n_pages} pages · {n_chunks} chunks")
                except Exception as e:
                    st.error(f"Error: {e}")

    st.markdown("---")

    # Status
    if st.session_state.pdf_ready:
        st.markdown(f'<div class="status-ready">✦ Ready — {st.session_state.pdf_name}</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-waiting">○ Waiting for PDF…</div>',
                    unsafe_allow_html=True)

    st.markdown("---")

    # Settings
    st.markdown("**Model**")
    model_choice = st.selectbox("", ["gpt-4o", "gpt-3.5-turbo"], label_visibility="collapsed")
    LLM_MODEL = model_choice

    if st.session_state.pdf_ready:
        if st.button("🗑 Clear Chat"):
            st.session_state.chat_history = []
            st.session_state.messages     = []
            st.rerun()


# ── Main chat area ────────────────────────────────────────────────────────────
if not st.session_state.pdf_ready:
    st.markdown("""
    <div style="display:flex; flex-direction:column; align-items:center;
                justify-content:center; height:60vh; text-align:center; opacity:0.4;">
        <div style="font-size:4rem; margin-bottom:1rem;">📄</div>
        <div style="font-family:'DM Serif Display',serif; font-size:1.5rem; color:#e8e3db;">
            Upload a PDF to get started
        </div>
        <div style="font-family:'DM Mono',monospace; font-size:0.75rem;
                    color:#c8a96e; margin-top:8px; letter-spacing:0.1em;">
            ADD KEY → UPLOAD PDF → PROCESS → CHAT
        </div>
    </div>
    """, unsafe_allow_html=True)

else:
    # Render chat messages
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-user">🧑 {msg["content"]}</div>',
                        unsafe_allow_html=True)
        else:
            sources_html = ""
            if msg.get("sources"):
                sources_html = f'<div class="chat-sources">📎 pages: {msg["sources"]}</div>'
            st.markdown(
                f'<div class="chat-assistant">🤖 {msg["content"]}{sources_html}</div>',
                unsafe_allow_html=True
            )

    # Chat input
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns([5, 1])
    with col1:
        question = st.text_input("", placeholder="Ask anything about your PDF…",
                                 label_visibility="collapsed", key="question_input")
    with col2:
        send = st.button("Send →")

    if (send or question) and question.strip():
        with st.spinner("Thinking…"):
            try:
                answer, sources = ask(
                    question,
                    st.session_state.collection,
                    st.session_state.openai_client,
                    st.session_state.chat_history,
                )
                st.session_state.messages.append({"role": "user",      "content": question})
                st.session_state.messages.append({"role": "assistant", "content": answer,
                                                   "sources": sources})
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")
