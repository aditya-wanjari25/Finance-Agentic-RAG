# FinSight RAG 🔍

A production-grade **Agentic RAG system** for financial document analysis, built on SEC 10-K filings.

## Architecture
```
PDF Documents (SEC 10-K/10-Q)
        ↓
┌───────────────────┐
│  Ingestion        │  PyMuPDF + pdfplumber → hierarchical chunking → OpenAI embeddings
│  Pipeline         │
└────────┬──────────┘
         ↓
┌───────────────────┐
│  Vector Store     │  ChromaDB with metadata filtering (ticker, year, section, page)
│  (ChromaDB)       │
└────────┬──────────┘
         ↓
┌───────────────────┐
│  LangGraph Agent  │  Query classification → filtered retrieval → GPT-4o generation
│                   │  Supports: retrieval, comparison, calculation, summarization
└────────┬──────────┘
         ↓
┌───────────────────┐
│  FastAPI          │  REST API with Pydantic validation + Swagger UI
└───────────────────┘
```

## Features

- **Finance-aware PDF parsing** — extracts tables, section headers, and narrative text from complex 10-K filings
- **Hierarchical chunking** — preserves table integrity, respects SEC section boundaries
- **Hybrid retrieval** — metadata filtering + semantic search for precise, company/year-specific results
- **Agentic reasoning** — LangGraph agent classifies queries and routes to specialized tools
- **Four query types** — retrieval, cross-period comparison, financial ratio calculation, section summarization
- **Structured answers** — every response includes citations with section and page number
- **RAGAS evaluation** — quantitative quality scoring with faithfulness, relevancy, precision and recall metrics

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | OpenAI GPT-4o |
| Orchestration | LangGraph |
| Vector Store | ChromaDB (local) |
| Embeddings | text-embedding-3-small |
| PDF Parsing | PyMuPDF + pdfplumber |
| API | FastAPI |
| Evaluation | RAGAS |
| Containerization | Docker |

## Quick Start

### Option 1 — Docker (recommended)
```bash
git clone https://github.com/YOUR_USERNAME/finsight-rag.git
cd finsight-rag
cp .env.example .env        # add your OPENAI_API_KEY
docker compose up -d
```

API available at `http://localhost:8000`
Swagger UI at `http://localhost:8000/docs`

### Option 2 — Local
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # add your OPENAI_API_KEY
uvicorn api.main:app --reload --port 8000
```

## Usage

### 1. Ingest a document
```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "pdf_path": "data/raw/AAPL_10K_2025.pdf",
    "ticker": "AAPL",
    "year": 2025
  }'
```

### 2. Query the agent
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the main risk factors for Apple?",
    "ticker": "AAPL",
    "year": 2025
  }'
```

### 3. Run evaluation
```bash
python -m evaluation.run_eval
```

## Evaluation Results

| Metric | Score |
|---|---|
| Faithfulness | 0.723 |
| Answer Relevancy | 0.859 |
| Context Precision | 0.544 |
| Context Recall | 0.450 |

*Evaluated on 10 financial Q&A pairs from AAPL 2025 10-K*


## Roadmap

- [ ] Hybrid retrieval (BM25 + semantic)
- [ ] Cross-document comparison (multiple tickers)
- [ ] Earnings call transcript ingestion
- [ ] AWS deployment (S3 + OpenSearch + ECS)
- [ ] Improved RAGAS scores through reranking