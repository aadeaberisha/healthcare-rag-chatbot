from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Optional, Tuple

import streamlit as st
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from rag.vectorstore import get_vectorstore
from rag.qa_chain import answer_question as rag_answer_question
from rag.config import (
    DEFAULT_MAX_DISTANCE,
    DEFAULT_TOP_K,
    NO_ANSWER,
    DEFAULT_PDF_DIR
)

load_dotenv()

st.set_page_config(
    page_title="Your Healthcare Assistant",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container{
        padding-top:1.2rem;
        max-width:980px;
        margin:0 auto;
        padding-left:0 !important;
        padding-right:0 !important;
      }

      html, body, [data-testid="stAppViewContainer"], .stApp{
        background:#ffffff !important;
        color:#0f172a !important;
      }

      h1,h2,h3,p,div,span,label{
        color:#0f172a !important;
      }

      h1{
        font-size:1.5rem !important;
        font-weight:600 !important;
      }

      .empty-wrap{ text-align:center; margin-top:20vh; }
      .empty-title{ font-size:2.1rem; font-weight:600; }
      .empty-sub{ font-size:0.95rem; opacity:0.7; margin-top:0.6rem; }

      .user-bubble{
        margin-left:auto;
        margin-right:0;
        background:#eef2ff;
        border-left:none;
        padding:14px 18px;
        border-radius:16px;
        width:fit-content;
        max-width:720px;
        box-shadow:none !important;
      }

      .bubble-label{
        font-size:0.72rem;
        font-weight:600;
        text-transform:uppercase;
        opacity:0.55;
        margin-bottom:0.3rem;
      }

      .assistant-plain{
        margin:0.8rem auto 1.4rem 0;
        max-width:820px;
        line-height:1.7;
        font-size:0.96rem;
      }

      .assistant-label{
        font-size:0.72rem;
        font-weight:600;
        text-transform:uppercase;
        opacity:0.45;
        margin-bottom:0.25rem;
      }

      section[data-testid="stSidebar"]{ width:340px !important; }

      div[data-testid="stChatInput"]{
        max-width:980px;
        margin:0 auto !important;
        padding-left:0 !important;
        padding-right:0 !important;
        transform:none !important;
      }

      div[data-testid="stChatInput"] textarea,
      div[data-testid="stChatInput"] input{
        width:100% !important;
      }

      hr[data-testid="stDivider"]{
        max-width:980px;
        margin-left:auto;
        margin-right:auto;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Your Healthcare Assistant")
st.caption("Answers are generated strictly from the uploaded healthcare documents. ")
st.divider()

_GREETING_WORDS = r"(hi|hello|hey|pershendetje|përshëndetje|tung|tungjatjeta)"
_GREETING_ONLY_RE = re.compile(
    rf"^\s*{_GREETING_WORDS}[\s!?.،,;:~\-–—_()\"'🙂🙋‍♀️🙋‍♂️😊👋]*\s*$",
    re.IGNORECASE,
)

def is_greeting(text: str) -> bool:
    return bool(_GREETING_ONLY_RE.match(text or ""))

def greeting_reply() -> str:
    return (
        "Hi! 👋\n"
        "Ask me something from your healthcare PDFs.\n\n"
        "Example:\n"
        "- What are the main use-cases of ML in operating room management?\n"
        "- Which algorithms are commonly used (e.g., XGBoost, Random Forest)?"
    )

def render_user(content: str):
    st.markdown(
        f"""
        <div class="user-bubble">
          <div class="bubble-label">You</div>
          {content.replace("\n","<br>")}
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_assistant(content: str):
    st.markdown(
        f"""
        <div class="assistant-plain">
          <div class="assistant-label">Assistant</div>
          {content.replace("\n","<br>")}
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_empty_state():
    st.markdown(
        """
        <div class="empty-wrap">
          <p class="empty-title">Where should we begin?</p>
          <p class="empty-sub">Ask anything about your healthcare documents.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

def list_documents(pdf_dir: str) -> list[str]:
    base = Path(pdf_dir)
    return sorted([p.name for p in base.glob("*.pdf")]) if base.exists() else []


def get_chunk_count(vectorstore) -> Optional[int]:
    """
    Returns the number of chunks stored in the FAISS docstore (if available).
    Safe for missing/older FAISS formats.
    """
    if vectorstore is None:
        return None
    try:
        store = getattr(vectorstore, "docstore", None)
        if store is None:
            return None
        d = getattr(store, "_dict", None)
        if isinstance(d, dict):
            return len(d)
        return None
    except Exception:
        return None


def kb_stats(pdf_dir: str, vectorstore) -> Tuple[int, Optional[int]]:
    num_docs = len(list_documents(pdf_dir))
    num_chunks = get_chunk_count(vectorstore)
    return num_docs, num_chunks


@st.cache_resource(show_spinner=False)
def cached_vectorstore():
    return get_vectorstore(rebuild=False)

def get_vs_if_ready(api_key: Optional[str]):
    if not api_key:
        return None
    return cached_vectorstore()


if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

if "source_filter" not in st.session_state:
    st.session_state.source_filter = None

if "last_question" not in st.session_state:
    st.session_state.last_question = None

if "last_answer" not in st.session_state:
    st.session_state.last_answer = None

if "memory_history" not in st.session_state:
    st.session_state.memory_history = []

api_key = os.getenv("OPENAI_API_KEY")
vectorstore = get_vs_if_ready(api_key)

with st.sidebar:
    st.header("Settings")

    if not api_key:
        st.warning("OPENAI_API_KEY missing.")

    st.divider()

    with st.expander("Knowledge base stats", expanded=True):
        num_docs, num_chunks = kb_stats(DEFAULT_PDF_DIR, vectorstore)
        st.caption(f"Documents: **{num_docs}**")
        if num_chunks is not None:
            st.caption(f"Chunks: **{num_chunks}**")
        else:
            st.caption("Chunks: **—** (index not loaded)")

    st.divider()

    new_chat = st.button("New chat", use_container_width=True)

    if new_chat:
        st.session_state.messages = []
        st.session_state.pending_question = None
        st.session_state.source_filter = None
        st.session_state.last_question = None
        st.session_state.last_answer = None
        st.session_state.doc_filter = "All documents"
        st.session_state.memory_history = []
        st.rerun()

    docs = ["All documents"] + list_documents(DEFAULT_PDF_DIR)
    selected_doc = st.selectbox("Filter by document", docs, index=0, key="doc_filter")

    if "prev_doc_filter" not in st.session_state:
        st.session_state.prev_doc_filter = selected_doc

    if selected_doc != st.session_state.prev_doc_filter:
        st.session_state.last_question = None
        st.session_state.last_answer = None
        st.session_state.prev_doc_filter = selected_doc
        st.session_state.memory_history = []
        st.rerun()

    st.session_state.source_filter = None if selected_doc == "All documents" else selected_doc


if not st.session_state.messages:
    render_empty_state()
else:
    for m in st.session_state.messages:
        if m["role"] == "user":
            render_user(m["content"])
        else:
            render_assistant(m["content"])
            if m.get("citations"):
                with st.expander("Sources"):
                    for c in m["citations"]:
                        st.markdown(f"- {c}")


pending_q = st.session_state.pending_question
if pending_q:
    st.session_state.pending_question = None

    if not api_key or vectorstore is None:
        st.session_state.messages.append(
            {"role": "assistant", "content": "Knowledge base is not ready.", "citations": []}
        )
        st.rerun()

    with st.spinner("Searching documents..."):
        MEMORY_TURNS = 3
        memory_text = "\n".join(st.session_state.memory_history[-MEMORY_TURNS:]).strip()

        res = rag_answer_question(
            question=pending_q,
            vectorstore=vectorstore,
            k=DEFAULT_TOP_K,
            max_distance=DEFAULT_MAX_DISTANCE,
            source_filter=st.session_state.source_filter,
            memory_text=memory_text,
        )

    answer = (res.answer or "").strip()
    citations = res.citations or []
    if answer == NO_ANSWER:
        citations = []

    st.session_state.last_question = pending_q
    st.session_state.last_answer = answer

    if answer != NO_ANSWER:
        st.session_state.memory_history.append(f"Q: {pending_q}\nA: {answer}")
    else:
        st.session_state.memory_history.append(f"Q: {pending_q}")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "citations": citations}
    )
    st.rerun()


question = st.chat_input("Ask anything")

if question:
    st.session_state.messages.append({"role": "user", "content": question})

    if is_greeting(question):
        st.session_state.messages.append(
            {"role": "assistant", "content": greeting_reply(), "citations": []}
        )
        st.session_state.last_question = None
        st.session_state.last_answer = None
        st.rerun()

    st.session_state.pending_question = question
    st.rerun()