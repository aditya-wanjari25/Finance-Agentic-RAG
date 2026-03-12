# Finance Agentic RAG

A production-grade Agentic RAG system for financial document analysis, built on SEC 10-K filings and earnings call transcripts.

## Tech Stack
- **LLM**: OpenAI GPT-4o
- **Orchestration**: LangGraph
- **Vector Store**: ChromaDB (local) → OpenSearch (AWS)
- **Embeddings**: OpenAI text-embedding-3-small
- **PDF Parsing**: PyMuPDF + pdfplumber
- **API**: FastAPI

## Project Status
- [x] Phase 1 — Ingestion Pipeline
- [x] Phase 2 — Agentic Layer
- [ ] Phase 3 — Production Hardening (in progress)
- [ ] Phase 4 — AWS Migration

## Setup
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # add your keys
```

## Usage
_Coming soon as phases are completed_
