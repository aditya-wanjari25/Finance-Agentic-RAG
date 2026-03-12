# api/routes/ingest.py

from fastapi import APIRouter, HTTPException
from api.schemas import IngestRequest, IngestResponse
from ingestion.pipeline import ingest_document

router = APIRouter()


@router.post("/ingest", response_model=IngestResponse)
async def ingest_pdf(request: IngestRequest):
    """
    Ingests a PDF document into the vector store.

    Runs the full pipeline:
    PDF → parse → chunk → embed → store in ChromaDB

    This is idempotent — running it twice on the same document
    will upsert (update) rather than duplicate chunks.

    Note: this is a long-running operation (30-60 seconds for a full 10-K).
    In production we'd run this as a background task via Celery or AWS SQS.
    For now it runs synchronously.
    """
    try:
        summary = ingest_document(
            pdf_path=request.pdf_path,
            ticker=request.ticker,
            year=request.year,
            quarter=request.quarter,
        )
        return IngestResponse(
            success=True,
            ticker=summary["ticker"],
            year=summary["year"],
            blocks_parsed=summary["blocks_parsed"],
            chunks_stored=summary["chunks_stored"],
            tokens_used=summary["tokens_used"],
            approximate_cost_usd=round(summary["tokens_used"] / 1_000_000 * 0.02, 4),
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")