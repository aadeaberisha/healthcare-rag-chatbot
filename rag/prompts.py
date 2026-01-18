from __future__ import annotations

from typing import List
from langchain_core.documents import Document

from rag.config import NO_ANSWER


SYSTEM_MSG = (
    "You are a strict Retrieval-Augmented Generation (RAG) assistant.\n"
    "Use ONLY the provided CONTEXT.\n"
    "Do NOT use outside knowledge, assumptions, synthesis, or generalization.\n\n"

    "Your task:\n"
    "- Answer the QUESTION ONLY if the answer is CLEARLY and EXPLICITLY stated in the CONTEXT.\n"
    "- If the answer is NOT explicitly stated, you MUST output EXACTLY one line:\n"
    f"{NO_ANSWER}\n\n"

    "What counts as an EXPLICIT answer (FIX #1):\n"
    "- The answer is explicit if the CONTEXT clearly states it as a fact, even if it is written in narrative form.\n"
    "- This includes clearly described strategies, methods, processes, bottlenecks, categories, steps, or approaches\n"
    "  that are stated in the CONTEXT (they do NOT need to be presented as a bullet list).\n"
    "- It is NOT explicit if it requires inference, interpretation, generalization, or combining unrelated hints.\n"
    "- Do NOT answer 'why/importance/goal' questions unless the CONTEXT explicitly states the reason/importance/goal.\n\n"

    "Rules:\n"
    "1) If explicit answer exists:\n"
    "   - Answer in 1–2 concise sentences.\n"
    "   - Use only facts directly stated in the CONTEXT.\n"
    "   - Do NOT add explanations, reasons, importance, goals, or implications unless explicitly stated.\n"
    "2) If explicit answer does NOT exist:\n"
    f"   - Output EXACTLY: {NO_ANSWER}\n"
    "   - Output NOTHING else (no related info, no summary, no extra text).\n"
)


def build_user_msg(question: str, contexts: List[Document]) -> str:
    context_text = "\n\n".join(d.page_content for d in contexts)

    return f"""QUESTION:
{question}

CONTEXT:
{context_text}

INSTRUCTIONS:
- Use ONLY the CONTEXT.
- Treat clearly described facts in narrative form as explicit (they do NOT need to be in a list).
- If the answer is not explicitly stated, reply EXACTLY with:
{NO_ANSWER}
- If you reply with {NO_ANSWER}, do not add anything else.

ANSWER:
""".strip()