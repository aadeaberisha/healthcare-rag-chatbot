from __future__ import annotations

from typing import List
from langchain_core.documents import Document


SYSTEM_MSG = (
    "You are a RAG assistant.\n"
    "Use ONLY the provided CONTEXT.\n"
    'If the answer is not explicitly stated in the context, reply exactly:\n'
    '"It is not explicitly stated in the documents."\n'
    "If there is related information in the context, summarize it and clearly label it as related (not an explicit definition).\n"
    "Do not guess or use outside knowledge.\n"
)


def build_user_msg(question: str, contexts: List[Document]) -> str:
    context_text = "\n\n".join(d.page_content for d in contexts)

    return f"""QUESTION:
{question}

CONTEXT:
{context_text}

RULES:
- Use ONLY the context.
- Do NOT use outside knowledge or assumptions.
- If the answer is not explicitly stated, start with: "It is not explicitly stated in the documents."
- If related information exists, add 1–3 sentences summarizing it and clearly state that it is related, not an explicit definition.
"""