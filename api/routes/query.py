# api/routes/query.py

from fastapi import APIRouter, HTTPException
from api.schemas import QueryRequest, QueryResponse, CitationResponse
from agents.graph import run_query

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def query_agent(request: QueryRequest):
    """
    Runs the FinSight agent against ingested documents.

    The agent will:
    1. Classify the query type (retrieval/comparison/calculation/summary)
    2. Retrieve relevant chunks with metadata filtering
    3. Generate a structured answer with citations

    Requires the requested ticker/year document to be ingested first.
    """
    try:
        result = run_query(
            query=request.query,
            ticker=request.ticker,
            year=request.year,
            quarter=request.quarter,
        )

        # Handle case where agent couldn't find relevant chunks
        if not result.get("final_answer"):
            raise HTTPException(
                status_code=404,
                detail=f"No answer generated. Ensure {request.ticker} "
                       f"{request.year} has been ingested."
            )

        # Build citation response objects
        citations = [
            CitationResponse(**c)
            for c in (result.get("citations") or [])
        ]

        return QueryResponse(
            answer=result["final_answer"],
            citations=citations,
            query_type=result.get("query_type", "retrieval"),
            ticker=request.ticker,
            year=request.year,
            chunks_retrieved=len(result.get("retrieved_chunks") or []),
        )

    except HTTPException:
        raise  # re-raise HTTP exceptions as-is
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Agent error: {str(e)}"
        )