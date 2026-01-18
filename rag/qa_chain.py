from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS

from rag.guardrails import is_prompt_injection

from rag.config import (
    NO_ANSWER,
    DEFAULT_LLM_MODEL,
    DEFAULT_MAX_SOURCES_SHORT,
    DEFAULT_MAX_SOURCES_LONG,
    DEFAULT_SHORT_ANSWER_CHAR_LIMIT,
)
from rag.prompts import SYSTEM_MSG, build_user_msg, REWRITE_QUERY_PROMPT
from rag.retriever import retrieve_with_scores, gate_and_select_contexts, build_citations


@dataclass
class RAGResult:
    answer: str
    citations: List[str]


def _rewrite_for_retrieval(
    question: str,
    memory_text: str,
    llm: ChatOpenAI,
) -> str:
    """
    Rewrites a follow-up question into a clear, standalone query for document search.

    Uses short-term conversation context only to resolve references (e.g., it/that/this).
    Does not add new information and is not used for answering the question.
    """

    if not memory_text.strip():
        return question

    prompt = REWRITE_QUERY_PROMPT.format(
        memory=memory_text,
        question=question,
    )

    try:
        resp = llm.invoke([("user", prompt)])
        rewritten = (resp.content or "").strip()
        return rewritten or question
    except Exception:
        return question


def answer_question(
    question: str,
    vectorstore: FAISS,
    k: int = 5,
    max_distance: float = 0.9,
    max_contexts: int = 5,
    source_filter: Optional[str] = None,
    model: str = DEFAULT_LLM_MODEL,
    temperature: float = 0.2,
    memory_text: str = "",
) -> RAGResult:
    if is_prompt_injection(question):
        return RAGResult(answer=NO_ANSWER, citations=[])
    
    llm = ChatOpenAI(model=model, temperature=temperature)

    retrieval_query = _rewrite_for_retrieval(question=question, memory_text=memory_text, llm=llm)

    docs_and_scores: List[Tuple[Document, float]] = retrieve_with_scores(
        retrieval_query,
        vectorstore,
        k=k,
        source_filter=source_filter,
    )

    contexts = gate_and_select_contexts(
        docs_and_scores,
        max_distance,
        max_contexts=max_contexts,
    )
    
    if not contexts and retrieval_query.strip() != question.strip():
        docs_and_scores = retrieve_with_scores(
            question,
            vectorstore,
            k=k,
            source_filter=source_filter,
        )
        contexts = gate_and_select_contexts(
            docs_and_scores,
            max_distance,
            max_contexts=max_contexts,
        )

    if not contexts:
        return RAGResult(answer=NO_ANSWER, citations=[])

    user_msg = build_user_msg(question, contexts)

    try:
        resp = llm.invoke([("system", SYSTEM_MSG), ("user", user_msg)])
        answer = (resp.content or "").strip().replace("\\n", "\n").strip()
    except Exception:
        answer = ""

    if not answer or answer.strip() == NO_ANSWER:
        return RAGResult(answer=NO_ANSWER, citations=[])

    max_sources = (
        DEFAULT_MAX_SOURCES_SHORT
        if len(answer) <= DEFAULT_SHORT_ANSWER_CHAR_LIMIT
        else DEFAULT_MAX_SOURCES_LONG
    )
    citations = build_citations(contexts, max_sources=max_sources)

    return RAGResult(answer=answer, citations=citations)