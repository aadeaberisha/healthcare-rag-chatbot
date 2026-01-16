from __future__ import annotations

from typing import List, Tuple

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS


def citation(d: Document) -> str:
    """
    Builds a user-facing citation string from a document chunk.

    The citation includes only:
    - document source (e.g. PDF filename)
    - page number (1-based)
    """
    src = d.metadata.get("source", "unknown")
    page = d.metadata.get("page")

    parts = [src]
    if page is not None:
        parts.append(f"p.{int(page) + 1}")

    return " | ".join(parts)


def retrieve_with_scores(
    question: str,
    vectorstore: FAISS,
    k: int = 5,
) -> List[Tuple[Document, float]]:
    """
    Performs a Top-K similarity search over the FAISS vector store.

    Returns:
        A list of (Document, score) tuples sorted by ascending distance.
        Lower scores indicate higher semantic similarity.
    """
    docs_and_scores: List[Tuple[Document, float]] = (
        vectorstore.similarity_search_with_score(question, k=k)
    )
    docs_and_scores.sort(key=lambda x: x[1])
    return docs_and_scores


def gate_and_select_contexts(
    docs_and_scores: List[Tuple[Document, float]],
    score_threshold: float,
) -> List[Document]:
    """
    Applies a relevance gate to retrieved documents to prevent hallucinations.

    If the best similarity score exceeds the threshold, no context is returned,
    forcing a safe "not in documents" response.

    Returns:
        A list of documents whose scores are within the relevance threshold.
    """
    if not docs_and_scores:
        return []

    best_score = docs_and_scores[0][1]
    if best_score > score_threshold:
        return []

    contexts = [d for (d, s) in docs_and_scores if s <= score_threshold]
    return contexts


def build_citations(contexts: List[Document]) -> List[str]:
    """
    Generates a list of unique citations from the selected context documents.

    Each citation follows the format:
        document_name | page_number

    Duplicate citations (e.g. multiple chunks from the same page)
    are removed for a cleaner presentation in the UI.
    """
    seen = set()
    citations: List[str] = []

    for d in contexts:
        c = citation(d)
        if c not in seen:
            seen.add(c)
            citations.append(c)

    return citations