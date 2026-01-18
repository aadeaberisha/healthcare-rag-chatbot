from __future__ import annotations

from pathlib import Path
from typing import Optional

from langchain_community.vectorstores import FAISS

from rag.config import DEFAULT_INDEX_DIR, DEFAULT_PDF_DIR
from rag.loaders import load_pdfs
from rag.chunking import chunk_documents
from rag.embeddings import get_embeddings


def get_vectorstore(
    rebuild: bool = False,
    pdf_dir: str = DEFAULT_PDF_DIR,
    index_dir: str = DEFAULT_INDEX_DIR,
    chunk_size: int = 900,
    chunk_overlap: int = 120,
) -> Optional[FAISS]:
    embeddings = get_embeddings()
    index_path = Path(index_dir)

    if index_path.exists() and not rebuild:
        return FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)

    docs = load_pdfs(pdf_dir)
    if not docs:
        return None

    chunks = chunk_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    vs = FAISS.from_documents(chunks, embeddings)
    index_path.mkdir(parents=True, exist_ok=True)
    vs.save_local(index_dir)
    return vs