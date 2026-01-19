from __future__ import annotations

from typing import List, Tuple, Optional

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
    source_filter: Optional[str] = None,
    oversample_k: int = 5,
) -> List[Tuple[Document, float]]:
    """
    Top-K search over FAISS.

    Note: FAISS returns distance scores (lower = more similar).
    If source_filter is provided, we oversample then filter by metadata["source"].
    """
    k_fetch = max(k * oversample_k, k)

    docs_and_scores: List[Tuple[Document, float]] = (
        vectorstore.similarity_search_with_score(question, k=k_fetch)
    )

    if source_filter:
        sf = source_filter.casefold()
        docs_and_scores = [
            (d, s)
            for (d, s) in docs_and_scores
            if str(d.metadata.get("source", "")).casefold() == sf
        ]

    docs_and_scores.sort(key=lambda x: x[1])
    return docs_and_scores[:k]


def gate_and_select_contexts(
    docs_and_scores: List[Tuple[Document, float]],
    max_distance: float,
    max_contexts: int = 5,
) -> List[Document]:
    """
    Applies a relevance gate to retrieved documents to prevent hallucinations.

    If the best similarity score exceeds the maximum allowed distance, no context is returned,
    forcing a safe "not in documents" response.

    Returns:
        A list of documents whose scores are within the relevance threshold.
    """
    if not docs_and_scores:
        return []

    best_score = docs_and_scores[0][1]
    if best_score > max_distance:
        return []

    contexts: List[Document] = []
    for d, s in docs_and_scores:
        if s <= max_distance:
            contexts.append(d)
        if len(contexts) >= max_contexts:
            break

    return contexts


def build_citations(
    contexts: List[Document],
    max_sources: int = 2,
) -> List[str]:
    """
    Generates a list of unique citations from the selected context documents.

    Each citation follows the format:
        document_name | page_number

    At most `max_sources` unique citations are returned to avoid clutter
    when the answer is short.
    """
    seen = set()
    citations: List[str] = []

    for d in contexts:
        c = citation(d)
        if c in seen:
            continue

        seen.add(c)
        citations.append(c)

        if len(citations) >= max_sources:
            break

    return citations