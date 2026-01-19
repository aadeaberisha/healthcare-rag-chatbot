from __future__ import annotations

NO_ANSWER = "It is not explicitly stated in the documents."

DEFAULT_INDEX_DIR = "data/processed/faiss_index"
DEFAULT_PDF_DIR = "data/raw_docs"

DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_LLM_MODEL = "gpt-4o-mini"

DEFAULT_TOP_K = 5
DEFAULT_MAX_DISTANCE = 1.1
DEFAULT_MAX_CONTEXTS = 5
DEFAULT_MAX_SOURCES_SHORT = 2
DEFAULT_MAX_SOURCES_LONG = 4
DEFAULT_SHORT_ANSWER_CHAR_LIMIT = 280