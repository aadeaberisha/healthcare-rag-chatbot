from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Optional

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

@st.cache_resource(show_spinner=False)
def cached_vectorstore(rebuild: bool = False):
    return get_vectorstore(rebuild=rebuild)

def get_vs_if_ready(api_key: Optional[str], rebuild: bool):
    if not api_key:
        return None, False
    vs = cached_vectorstore(rebuild=rebuild)
    return vs, (vs is not None)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

with st.sidebar:
    st.header("Settings")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.warning("OPENAI_API_KEY missing.")

    st.divider()

    col1, col2 = st.columns(2)
    new_chat = col1.button("New chat", use_container_width=True)
    rebuild_index = col2.button("Rebuild", use_container_width=True)

    if new_chat:
        st.session_state.messages = []
        st.session_state.pending_question = None
        st.rerun()

vectorstore, index_ready = get_vs_if_ready(api_key, rebuild=rebuild_index)

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

    if not api_key or not index_ready or vectorstore is None:
        st.session_state.messages.append(
            {"role": "assistant", "content": "Knowledge base is not ready.", "citations": []}
        )
        st.rerun()

    with st.spinner("Searching documents..."):
        res = rag_answer_question(
            question=pending_q,
            vectorstore=vectorstore,
            k=DEFAULT_TOP_K,
            max_distance=DEFAULT_MAX_DISTANCE,
        )

        answer = (res.answer or "").strip()
        citations = res.citations or []

        if answer == NO_ANSWER:
            citations = []

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
        st.rerun()

    st.session_state.pending_question = question
    st.rerun()