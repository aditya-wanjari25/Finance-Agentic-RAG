# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run the API server locally
uvicorn api.main:app --reload --port 8000

# Run with Docker (recommended — persists ChromaDB via volume mount)
docker compose up -d
docker compose logs -f

# Ingest a document
python -m ingestion.pipeline   # runs __main__ block (AAPL 2025 example)

# Run RAGAS evaluation
python -m evaluation.run_eval

# AWS deployment
bash scripts/aws-up.sh      # provision and deploy
bash scripts/aws-status.sh  # check ECS task/service status
bash scripts/aws-down.sh    # tear down infrastructure
```

There is no test suite currently — `tests/` is empty.

## Architecture

The system has four stages that form a linear pipeline:

```
PDF → Ingestion Pipeline → Vector Store → LangGraph Agent → FastAPI
```

### 1. Ingestion (`ingestion/`)

`pipeline.py:ingest_document()` is the entry point. It chains:
- **Parser** (`parsers/pdf_parser.py`) — PyMuPDF + pdfplumber extract `ParsedBlock` objects tagged as `text` or `table`, with section headers detected via font size/weight heuristics.
- **Chunker** (`chunkers/hierarchical_chunker.py`) — Produces `Chunk` dataclasses. Text chunks cap at 512 tokens with 50-token overlap; tables are kept whole (up to 1024 tokens). Each chunk gets a content prefix (e.g., `"AAPL 2025 Risk Factors: ..."`) for embedding quality, plus metadata: `ticker`, `year`, `section`, `page`, `chunk_type`.
- **Embedder** (`embedders/openai_embedder.py`) — `text-embedding-3-small` via OpenAI API.
- **Store** — writes to `VectorStore` (ChromaDB) or `OpenSearchStore` depending on env.

### 2. Retrieval (`retrieval/`)

`vector_store.get_vector_store()` is a factory — returns `VectorStore` (ChromaDB, local) or `OpenSearchStore` (AWS), controlled by `USE_OPENSEARCH=true` env var. Both expose the same interface: `.query()`, `.get_by_metadata()`, `.store()`.

`reranker.py` provides an optional cross-encoder rerank step (`cross-encoder/ms-marco-MiniLM-L-6-v2`, runs locally on CPU). Pattern: retrieve 20 candidates → rerank → return top 6.

### 3. Multi-Agent System (`agents/`)

The system uses a **supervisor + specialists** pattern. The public entry point is `agents/graph.py:run_query()`, which is unchanged from the API's perspective.

**Supervisor graph** (`agents/supervisor.py`):
```
classify → [route] → {specialist agent} → END
                  ↘ handle_error → END
```
The `classify` node uses GPT-4o to set `query_type`, `section_filter`, `comparison_year`. Cross-company detection (via `TICKER_MAP`) runs first and overrides the LLM classification.

**Specialist agents** (`agents/specialists/`) — each is an independent LangGraph subgraph:
```
retrieve → generate → END
```
| Specialist | Query type | Retrieval strategy |
|---|---|---|
| `RetrievalAgent` | `retrieval` | Semantic search, top-5 |
| `ComparisonAgent` | `comparison` | `ComparePeriodsTool` — two-year retrieval |
| `CalculationAgent` | `calculation` | Semantic search top-6 + `CalculateRatioTool` |
| `SummarizationAgent` | `summary` | `get_by_metadata` — all section chunks |
| `CrossCompanyAgent` | `cross_company` | Semantic search for each ticker |

**State types** (`agents/state.py`):
- `SupervisorState` — flows through the supervisor graph; holds routing fields + final outputs
- `SpecialistState` — internal to each specialist subgraph; same shape but includes `tool_results`
- `AgentState` — legacy, kept in `nodes.py` which is no longer active

Each specialist's `invoke(SupervisorState) -> dict` transforms supervisor state to specialist state via `specialists/base.py:specialist_input()`, runs the subgraph, and returns `final_answer + citations + retrieved_chunks`.

**Shared utilities** (`agents/specialists/base.py`) — `client` (single OpenAI instance), `extract_citations()`, `format_chunks()`, `specialist_input()`.

**Prompts** live in `agents/prompts/` — `system.py` (SYSTEM_PROMPT), `templates.py` (QUERY_ANALYSIS_TEMPLATE, GENERATION_TEMPLATE, COMPARISON_TEMPLATE, CROSS_COMPANY_TEMPLATE).

### 4. API (`api/`)

FastAPI with three routers: `POST /ingest`, `POST /query`, `GET /health`. Pydantic schemas in `api/schemas.py`. The public entry point for the agent is `agents.graph.run_query(query, ticker, year, quarter)`.

### Observability

LangSmith tracing is opt-in via env vars. `agents/observability.py:get_langsmith_config()` returns a config dict (or empty dict if disabled) that's passed to `agent.invoke()`. Enable with `LANGCHAIN_TRACING_V2=true` + `LANGSMITH_API_KEY`.

## Environment Variables

| Variable | Purpose |
|---|---|
| `OPENAI_API_KEY` | Required — embeddings + GPT-4o |
| `USE_OPENSEARCH` | `true` to use AWS OpenSearch instead of ChromaDB |
| `OPENSEARCH_HOST` | OpenSearch endpoint URL |
| `AWS_REGION` | AWS region for S3/OpenSearch |
| `S3_BUCKET_NAME` | Bucket for PDF storage |
| `LANGCHAIN_TRACING_V2` | `true` to enable LangSmith tracing |
| `LANGSMITH_API_KEY` | LangSmith API key |

ChromaDB persists to `.chroma/` in the project root (mounted as a Docker volume in `docker-compose.yml`).

## Key Design Decisions

- **Metadata-filtered retrieval**: every query filters by `ticker` + `year` before semantic search, preventing cross-company/cross-year leakage in a multi-document store.
- **Chunk IDs** are deterministic: `{ticker}_{year}_chunk{index}` — upsert is idempotent, re-ingesting a document updates existing chunks.
- **`get_vector_store()` factory** is the single switch between local (ChromaDB) and production (OpenSearch) — all agent tools call this, never instantiate stores directly.
- **`comparison_ticker` detection** uses a hardcoded `TICKER_MAP` (6 companies) in `agents/supervisor.py`. Adding a new company requires updating this dict and ensuring the company's documents are ingested.
- **Specialist isolation** — each specialist agent has its own `StateGraph` instance. They share stateless tool objects (`RetrieveTool`, etc.) but their graph state is fully independent, making them safe to extend or replace without touching other agents.
