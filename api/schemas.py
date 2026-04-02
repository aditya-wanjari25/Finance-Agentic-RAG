# api/schemas.py

from pydantic import BaseModel, Field
from typing import Optional, Any
from enum import Enum


class QuarterEnum(str, Enum):
    annual = "annual"
    q1 = "Q1"
    q2 = "Q2"
    q3 = "Q3"


class QueryRequest(BaseModel):
    """
    Request body for POST /query.
    Pydantic validates types and required fields automatically.
    Field() lets us add descriptions that appear in /docs.
    """
    query: str = Field(
        ...,
        description="The financial question to answer",
        example="What are the main risk factors for Apple in 2025?"
    )
    ticker: str = Field(
        ...,
        description="Stock ticker symbol",
        example="AAPL"
    )
    year: int = Field(
        ...,
        description="Fiscal year of the filing",
        example=2025
    )
    quarter: QuarterEnum = Field(
        default=QuarterEnum.annual,
        description="Filing period"
    )

    class Config:
        # Allows the enum to serialize as its string value
        use_enum_values = True


class CitationResponse(BaseModel):
    ticker: str
    year: int
    section: str
    page: int
    chunk_type: str


class QueryResponse(BaseModel):
    """Response body for POST /query."""
    answer: str
    citations: list[CitationResponse]
    query_type: str
    ticker: str
    year: int
    chunks_retrieved: int


class IngestRequest(BaseModel):
    """Request body for POST /ingest."""
    pdf_path: str = Field(
        ...,
        description="Absolute or relative path to the PDF file on the server",
        example="data/raw/AAPL_10K_2025.pdf"
    )
    ticker: str = Field(..., example="AAPL")
    year: int = Field(..., example=2025)
    quarter: QuarterEnum = Field(default=QuarterEnum.annual)

    class Config:
        use_enum_values = True


class IngestResponse(BaseModel):
    """Response body for POST /ingest."""
    success: bool
    ticker: str
    year: int
    blocks_parsed: int
    chunks_stored: int
    tokens_used: int
    approximate_cost_usd: float


class HealthResponse(BaseModel):
    """Response body for GET /health."""
    status: str
    vector_store_chunks: int
    collection_name: str
    openai_configured: bool
    langsmith_tracing: bool 