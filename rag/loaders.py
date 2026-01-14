from __future__ import annotations

import re
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


def clean_text(text: str) -> str:
    """
    Minimal, safe text cleaning:
    - normalize extra newlines
    - normalize multiple spaces
    - remove common page markers
    """
    text = re.sub(r"\n{3,}", "\n\n", text)

    text = re.sub(r"[ \t]{2,}", " ", text)

    text = re.sub(r"Page\s+\d+\s+of\s+\d+", "", text, flags=re.IGNORECASE)

    return text.strip()


def load_pdfs(folder: str = "data/raw_docs") -> List[Document]:
    """
    Loads all PDFs from a folder as LangChain Documents.
    Each PDF is split per page and includes source filename and 0-based page index.
    """
    base = Path(folder)
    if not base.exists():
        raise FileNotFoundError(f"Folder not found: {base.resolve()}")

    pdfs = sorted(base.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDFs found in: {base.resolve()}")

    docs: List[Document] = []
    for path in pdfs:
        loader = PyPDFLoader(str(path))
        pages = loader.load()

        for d in pages:
            d.metadata["source"] = path.name
            d.page_content = clean_text(d.page_content)

        docs.extend(pages)

    return docs