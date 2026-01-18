from __future__ import annotations

from langchain_openai import OpenAIEmbeddings
from rag.config import DEFAULT_EMBEDDING_MODEL


def get_embeddings(model: str = DEFAULT_EMBEDDING_MODEL) -> OpenAIEmbeddings:
    """
    OpenAI embeddings client.
    """
    return OpenAIEmbeddings(model=model)