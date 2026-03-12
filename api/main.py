# api/main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import query, ingest, health

app = FastAPI(
    title="FinSight RAG API",
    description=(
        "Production-grade Agentic RAG system for financial document analysis. "
        "Supports SEC 10-K and 10-Q filings with multi-document comparison, "
        "financial ratio calculation, and section summarization."
    ),
    version="0.1.0",
    docs_url="/docs",       # Swagger UI at /docs
    redoc_url="/redoc",     # ReDoc UI at /redoc
)

# CORS — allows frontend apps to call this API
# In production, replace "*" with your actual frontend domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(health.router, tags=["System"])
app.include_router(ingest.router, tags=["Ingestion"])
app.include_router(query.router, tags=["Agent"])


@app.get("/", tags=["System"])
async def root():
    return {
        "service": "FinSight RAG API",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health",
    }