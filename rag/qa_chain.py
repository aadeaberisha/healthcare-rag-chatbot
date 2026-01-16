from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS

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
    score_threshold: float = 0.9,
    model: str = "gpt-4o-mini",
    temperature: float = 0.2,
) -> RAGResult:
    docs_and_scores: List[Tuple[Document, float]] = retrieve_with_scores(question, vectorstore, k=k)

    if not docs_and_scores:
        return RAGResult(
            answer="It is not explicitly stated in the documents.",
            citations=[],
        )

    contexts = gate_and_select_contexts(docs_and_scores, score_threshold)

    if not contexts:
        return RAGResult(
            answer="It is not explicitly stated in the documents.",
            citations=[],
        )

    citations = build_citations(contexts)
    user_msg = build_user_msg(question, contexts)

    llm = ChatOpenAI(model=model, temperature=temperature)

    try:
        resp = llm.invoke([("system", SYSTEM_MSG), ("user", user_msg)])
        answer = (resp.content or "").strip().replace("\\n", "\n").strip()
    except Exception:
        answer = ""

    if not answer:
        answer = (
            "Relevant information was found in the documents, "
            "but an answer could not be generated at this time."
        )

    return RAGResult(answer=answer, citations=citations)