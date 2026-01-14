from __future__ import annotations

from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(
    docs: List[Document],
    chunk_size: int = 900,
    chunk_overlap: int = 120,
) -> List[Document]:
    """
    Splits documents into chunks while keeping metadata (source/page).
    Adds a chunk_index per source to make citations cleaner.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )

    chunks = splitter.split_documents(docs)

    per_source_counter = {}
    for c in chunks:
        src = c.metadata.get("source", "unknown")
        per_source_counter[src] = per_source_counter.get(src, 0) + 1
        c.metadata["chunk_index"] = per_source_counter[src]

    return chunks
