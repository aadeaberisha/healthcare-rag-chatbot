# Healthcare RAG Chatbot (Project #4)

A strict Retrieval-Augmented Generation (RAG) chatbot for healthcare PDFs that answers questions **only** from retrieved document context, with citations and safety guardrails.

## Overview

This project implements a grounded RAG system that:
- Answers questions **strictly** from retrieved context—no synthesis, generalization, or outside knowledge
- Returns exactly `"It is not explicitly stated in the documents."` when answers are not explicitly found
- Provides citations with document name and page number

## Features

- **Vector Search**: FAISS Top-K retrieval with cosine similarity
- **Citations**: Each answer includes source citations with document name and page number
- **NO_ANSWER Safety**: Returns exactly `"It is not explicitly stated in the documents."` when context doesn't contain explicit answers
- **Streamlit Web App**: Clean, modern chat interface with document filtering and knowledge base stats

## Project Structure

```
healthcare-rag-chatbot/
├── app/
│   └── streamlit_app.py          # Streamlit web application
├── rag/
│   ├── chunking.py                # Document chunking with metadata
│   ├── config.py                  # Configuration constants
│   ├── embeddings.py              # OpenAI embedding model setup
│   ├── guardrails.py              # Prompt injection detection
│   ├── loaders.py                 # PDF loading utilities
│   ├── prompts.py                 # System and user prompts
│   ├── qa_chain.py                # Main RAG answer generation logic
│   ├── retriever.py               # FAISS retrieval and citation building
│   └── vectorstore.py             # FAISS index loading/rebuilding
├── data/
│   ├── raw_docs/                  # Source PDF files
│   └── processed/
│       └── faiss_index/            # FAISS vector index
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## Setup

### Prerequisites
- Python 3.8+
- OpenAI API key

### Installation

1. Create a virtual environment:
   ```bash
   python -m venv venv
   ```

2. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - macOS/Linux: `source venv/bin/activate`

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## Run Locally

Start the Streamlit application:
```bash
streamlit run app/streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`.

## How It Works (High Level)

1. The user submits a healthcare-related question
2. The retriever finds the Top-K most relevant chunks
3. The LLM generates an answer strictly from retrieved context
4. Sources are shown with document name + page number
5. If the answer is not found → "It is not explicitly stated in the documents."

## Example Questions

- Does prior authorization increase time to treatment initiation for cancer drugs?
- What range of AUC values was reported for AI models using HIE data?
- In which healthcare settings are patient prioritization tools used according to the review?
- Which model achieved higher accuracy compared to the senior billing coder?
