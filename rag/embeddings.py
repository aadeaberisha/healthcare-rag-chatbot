from __future__ import annotations

from langchain_openai import OpenAIEmbeddings


def get_embeddings(model: str = "text-embedding-3-small") -> OpenAIEmbeddings:
    """
    OpenAI embeddings client.
    """
    return OpenAIEmbeddings(model=model)