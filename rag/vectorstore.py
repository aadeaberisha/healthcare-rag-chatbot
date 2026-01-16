from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

DEFAULT_INDEX_DIR = "data/processed/faiss_index"


def load_index(
    embeddings: Embeddings,
    index_dir: str = DEFAULT_INDEX_DIR,
) -> Optional[FAISS]:
    """
    Load FAISS index if it exists.
    """
    p = Path(index_dir)
    if not p.exists():
        return None
    return FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)


def build_and_save_index(
    chunks: List[Document],
    embeddings: Embeddings,
    index_dir: str = DEFAULT_INDEX_DIR,
) -> FAISS:
    """
    Build a FAISS index from chunks and save it to disk.
    """
    vs = FAISS.from_documents(chunks, embeddings)
    Path(index_dir).mkdir(parents=True, exist_ok=True)
    vs.save_local(index_dir)
    return vs