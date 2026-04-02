# api/routes/health.py

import os
from fastapi import APIRouter
from api.schemas import HealthResponse
from retrieval.vector_store import get_vector_store
from agents.observability import is_tracing_enabled

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    try:
        store = get_vector_store()
        stats = store.get_stats()
        vector_store_chunks = stats["total_chunks"]
        collection_name = stats.get("collection", stats.get("index", "unknown"))
    except Exception:
        vector_store_chunks = -1
        collection_name = "unavailable"

    return HealthResponse(
        status="healthy",
        vector_store_chunks=vector_store_chunks,
        collection_name=collection_name,
        openai_configured=bool(os.getenv("OPENAI_API_KEY")),
        langsmith_tracing=is_tracing_enabled(),
    )