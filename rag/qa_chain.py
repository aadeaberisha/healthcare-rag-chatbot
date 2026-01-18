from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS

from rag.config import (
    NO_ANSWER,
    DEFAULT_LLM_MODEL,
    DEFAULT_MAX_SOURCES_SHORT,
    DEFAULT_MAX_SOURCES_LONG,
    DEFAULT_SHORT_ANSWER_CHAR_LIMIT,
)
from rag.prompts import SYSTEM_MSG, build_user_msg
from rag.retriever import retrieve_with_scores, gate_and_select_contexts, build_citations


@dataclass
class RAGResult:
    answer: str
    citations: List[str]


def answer_question(
    question: str,
    vectorstore: FAISS,
    k: int = 5,
    max_distance: float = 0.9,
    max_contexts: int = 5,
    model: str = DEFAULT_LLM_MODEL,
    temperature: float = 0.2,
) -> RAGResult:
    docs_and_scores: List[Tuple[Document, float]] = retrieve_with_scores(question, vectorstore, k=k)

    contexts = gate_and_select_contexts(
        docs_and_scores,
        max_distance,
        max_contexts=max_contexts,
    )
    if not contexts:
        return RAGResult(answer=NO_ANSWER, citations=[])

    user_msg = build_user_msg(question, contexts)
    llm = ChatOpenAI(model=model, temperature=temperature)

    try:
        resp = llm.invoke([("system", SYSTEM_MSG), ("user", user_msg)])
        answer = (resp.content or "").strip().replace("\\n", "\n").strip()
    except Exception:
        answer = ""

    if not answer:
        return RAGResult(answer=NO_ANSWER, citations=[])

    if answer.strip() == NO_ANSWER:
        return RAGResult(answer=NO_ANSWER, citations=[])

    max_sources = (
        DEFAULT_MAX_SOURCES_SHORT
        if len(answer) <= DEFAULT_SHORT_ANSWER_CHAR_LIMIT
        else DEFAULT_MAX_SOURCES_LONG
    )
    citations = build_citations(contexts, max_sources=max_sources)

    return RAGResult(answer=answer, citations=citations)