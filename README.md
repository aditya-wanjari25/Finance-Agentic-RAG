# FinSight RAG 🔍

A production-grade **Agentic RAG system** for financial document analysis, built on SEC 10-K filings and deployed on AWS.

## Architecture
```
PDF Documents (SEC 10-K/10-Q)
        ↓
┌───────────────────┐
│  Ingestion        │  PyMuPDF + pdfplumber → hierarchical chunking → OpenAI embeddings
│  Pipeline         │
└────────┬──────────┘
         ↓
┌─────────────────────────────────────┐
│  Storage Layer                      │
│  AWS S3 (documents)                 │
│  AWS OpenSearch (vectors + BM25)    │
└────────┬────────────────────────────┘
         ↓
┌───────────────────┐
│  LangGraph Agent  │  Query classification → filtered retrieval → GPT-4o generation
│                   │  Tools: retrieve, compare, calculate, summarize
└────────┬──────────┘
         ↓
┌───────────────────┐
│  FastAPI          │  REST API with Pydantic validation + Swagger UI
└────────┬──────────┘
         ↓
┌───────────────────┐
│  AWS ECS Fargate  │  Containerized deployment with auto-restart
└───────────────────┘
```

## Features

- **Finance-aware PDF parsing** — extracts tables, section headers, and narrative text from complex 10-K filings using PyMuPDF + pdfplumber
- **Hierarchical chunking** — preserves table integrity, respects SEC section boundaries, filters table-of-contents noise
- **Metadata-filtered retrieval** — every chunk tagged with ticker, year, section, page — enables precise company/year-specific search
- **Agentic reasoning** — LangGraph agent classifies queries and routes to specialized financial tools
- **Four query types** — retrieval, cross-period comparison, financial ratio calculation, section summarization
- **Cross-company comparison** — compare any two ingested companies in a single query e.g. "Compare Apple and Google's risk factors"
- **Structured answers** — every response includes citations with section and page number
- **RAGAS evaluation** — quantitative quality scoring with faithfulness, relevancy, precision and recall
- **LangSmith Tracing** - Added end-to-end observability, enabling run-level metrics (latency, cost, tags) and node-level tracing to debug retrieval vs generation issues in production.
- **Production AWS deployment** — ECS Fargate + OpenSearch + S3 + Secrets Manager

## Tech Stack

| Layer | Local | Production (AWS) |
|---|---|---|
| LLM | OpenAI GPT-4o | OpenAI GPT-4o |
| Orchestration | LangGraph | LangGraph |
| Vector Store | ChromaDB | AWS OpenSearch |
| Document Storage | Local filesystem | AWS S3 |
| Embeddings | text-embedding-3-small | text-embedding-3-small |
| PDF Parsing | PyMuPDF + pdfplumber | PyMuPDF + pdfplumber |
| Observability | LangSmith | LangSmith
| API | FastAPI | FastAPI |
| Container | Docker | AWS ECS Fargate |
| Secrets | .env file | AWS Secrets Manager |
| Evaluation | RAGAS | RAGAS |


## AWS Infrastructure
```
ECR          — Private Docker image registry
ECS Fargate  — Serverless container runtime (1 vCPU, 2GB RAM)
OpenSearch   — Managed vector + keyword search (t3.small)
S3           — Document storage with versioning
Secrets Manager — Encrypted API key storage
CloudWatch   — Container logging and monitoring
```

## Quick Start

### Option 1 — Docker (recommended)
```bash
git clone https://github.com/YOUR_USERNAME/finsight-rag.git
cd finsight-rag
cp .env.example .env        # add your OPENAI_API_KEY
docker compose up -d
```

API at `http://localhost:8000` — Swagger UI at `http://localhost:8000/docs`

### Option 2 — Local
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
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
# Retrieval query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the main risk factors for Apple?",
    "ticker": "AAPL",
    "year": 2025
  }'

# Calculation query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is Apples gross margin for 2025?",
    "ticker": "AAPL",
    "year": 2025
  }'

# Summary query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Summarize the MD&A section",
    "ticker": "AAPL",
    "year": 2025
  }'

# Cross-company comparison
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Compare Apple and Google risk factors",
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

- [ ] Application Load Balancer (stable DNS endpoint)
- [ ] Query expansion for improved context recall
- [ ] Hybrid BM25 + vector search in OpenSearch
- [ ] Earnings call transcript ingestion
- [ ] Lambda trigger on S3 upload for auto-ingestion