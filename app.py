"""
DocuChat AI - PDF Q&A Chatbot
Streamlit UI for a LangChain + Groq + HuggingFace + ChromaDB RAG pipeline.
"""

import os
import tempfile

import streamlit as st

from rag_pipeline import build_knowledge_base, ask


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="DocuChat AI",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# CUSTOM CSS (Light theme, blue + orange accents)
# =========================================================
def load_css():
    st.markdown(
        """
        <style>
            :root {
                --primary: #2563EB;
                --accent: #F97316;
                --bg: #F8FAFC;
            }

            .stApp {
                background-color: var(--bg);
            }

            /* Header */
            .app-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 0.75rem 1rem;
                background: white;
                border-radius: 14px;
                box-shadow: 0 1px 6px rgba(0,0,0,0.06);
                margin-bottom: 1rem;
            }
            .app-header h1 {
                font-size: 1.3rem;
                margin: 0;
                color: #0F172A;
            }
            .app-header span {
                color: #64748B;
                font-size: 0.85rem;
            }

            /* Sidebar cards */
            section[data-testid="stSidebar"] {
                background-color: #FFFFFF;
            }
            .sidebar-card {
                background: white;
                border-radius: 12px;
                padding: 0.9rem;
                box-shadow: 0 1px 4px rgba(0,0,0,0.05);
                margin-bottom: 0.9rem;
            }
            .status-ready {
                color: #16A34A;
                font-weight: 600;
                background: #DCFCE7;
                padding: 4px 10px;
                border-radius: 8px;
                font-size: 0.85rem;
                display: inline-block;
            }
            .status-processing {
                color: #B45309;
                font-weight: 600;
                background: #FEF3C7;
                padding: 4px 10px;
                border-radius: 8px;
                font-size: 0.85rem;
                display: inline-block;
            }
            .status-empty {
                color: #64748B;
                font-weight: 600;
                background: #F1F5F9;
                padding: 4px 10px;
                border-radius: 8px;
                font-size: 0.85rem;
                display: inline-block;
            }
            .stat-row {
                display: flex;
                justify-content: space-between;
                font-size: 0.9rem;
                color: #334155;
                padding: 3px 0;
            }

            /* Source tag under AI answers */
            .source-tag {
                display: inline-block;
                margin-top: 6px;
                background: #FFF7ED;
                color: var(--accent);
                border: 1px solid #FED7AA;
                border-radius: 8px;
                padding: 2px 10px;
                font-size: 0.8rem;
            }

            /* Footer */
            .app-footer {
                text-align: center;
                color: #94A3B8;
                font-size: 0.8rem;
                padding: 1rem 0 0.4rem 0;
            }

            /* Buttons */
            .stButton>button {
                border-radius: 10px;
                font-weight: 600;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# SESSION STATE INIT
# =========================================================
def init_session_state():
    defaults = {
        "vector_store": None,
        "messages": [],          # [{"role": "user"/"assistant", "content": str, "sources": list}]
        "uploaded_names": [],
        "status": "empty",       # "empty" | "processing" | "ready"
        "stats": {"pdfs": 0, "pages": 0, "chunks": 0},
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# =========================================================
# SIDEBAR
# =========================================================
def render_sidebar():
    with st.sidebar:
        st.markdown("### 📄 DocuChat AI")
        st.caption("Chat with your PDFs using AI")

        st.markdown("---")

        uploaded_files = st.file_uploader(
            "📁 Upload PDF(s)",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload one or more PDF files to build your knowledge base.",
        )

        if uploaded_files:
            st.markdown("**Uploaded files:**")
            for f in uploaded_files:
                st.markdown(f"✔ {f.name}")

        build_clicked = st.button(
            "▶ Build Knowledge Base",
            use_container_width=True,
            disabled=not uploaded_files,
        )

        if build_clicked and uploaded_files:
            process_pdfs(uploaded_files)

        st.markdown("---")

        # Status badge
        status = st.session_state.status
        if status == "ready":
            st.markdown('<span class="status-ready">🟢 Ready</span>', unsafe_allow_html=True)
        elif status == "processing":
            st.markdown('<span class="status-processing">🟡 Processing</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-empty">⚪ No PDFs yet</span>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Stats
        stats = st.session_state.stats
        st.markdown(
            f"""
            <div class="sidebar-card">
                <div class="stat-row"><span>📊 PDFs</span><b>{stats['pdfs']}</b></div>
                <div class="stat-row"><span>📄 Pages</span><b>{stats['pages']}</b></div>
                <div class="stat-row"><span>🧩 Chunks</span><b>{stats['chunks']}</b></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button("🗑 Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


# =========================================================
# PDF PROCESSING
# =========================================================
def process_pdfs(uploaded_files):
    st.session_state.status = "processing"

    with st.spinner("Processing PDFs and building vector database..."):
        try:
            # Save uploaded files to temp paths (loaders expect file paths)
            tmp_paths = []
            with tempfile.TemporaryDirectory() as tmp_dir:
                for f in uploaded_files:
                    path = os.path.join(tmp_dir, f.name)
                    with open(path, "wb") as out:
                        out.write(f.getbuffer())
                    tmp_paths.append(path)

                vector_store = build_knowledge_base(tmp_paths)

            st.session_state.vector_store = vector_store
            st.session_state.uploaded_names = [f.name for f in uploaded_files]

            # Best-effort stats (falls back gracefully if internals differ)
            try:
                collection = vector_store._collection
                chunk_count = collection.count()
            except Exception:
                chunk_count = 0

            st.session_state.stats = {
                "pdfs": len(uploaded_files),
                "pages": "—",
                "chunks": chunk_count,
            }
            st.session_state.status = "ready"
            st.success(f"Knowledge base built from {len(uploaded_files)} PDF(s). Ask away!")

        except Exception as e:
            st.session_state.status = "empty"
            st.error(f"Failed to process PDFs: {e}")


# =========================================================
# CHAT AREA
# =========================================================
def render_header():
    st.markdown(
        """
        <div class="app-header">
            <h1>📄 DocuChat AI</h1>
            <span>🔍 Chat with your PDFs</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_chat_history():
    if not st.session_state.messages:
        st.info("👋 Upload a PDF from the sidebar and build the knowledge base to start chatting.")
        return

    for msg in st.session_state.messages:
        role = msg["role"]
        avatar = "👤" if role == "user" else "🤖"
        with st.chat_message(role, avatar=avatar):
            st.markdown(msg["content"])
            if role == "assistant" and msg.get("sources"):
                pages = ", ".join(str(p) for p in msg["sources"])
                st.markdown(
                    f'<span class="source-tag">📄 Source: Page {pages}</span>',
                    unsafe_allow_html=True,
                )


def handle_user_input():
    is_ready = st.session_state.status == "ready" and st.session_state.vector_store is not None

    placeholder = "Ask anything about your PDFs..." if is_ready else "Build the knowledge base first ⬅️"
    question = st.chat_input(placeholder, disabled=not is_ready)

    if question:
        st.session_state.messages.append({"role": "user", "content": question, "sources": []})

        with st.spinner("Thinking..."):
            try:
                answer, sources = ask(question, st.session_state.vector_store)
            except Exception as e:
                answer, sources = f"⚠️ Error while generating answer: {e}", []

        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )
        st.rerun()


def render_footer():
    st.markdown(
        """
        <div class="app-footer">
            Powered by Innovative-Minds12
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# MAIN
# =========================================================
def main():
    load_css()
    init_session_state()

    render_sidebar()
    render_header()
    render_chat_history()
    handle_user_input()
    render_footer()


if __name__ == "__main__":
    main()
