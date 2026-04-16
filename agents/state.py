# agents/state.py

from typing import TypedDict, Optional, Any


class RetrievedChunk(TypedDict):
    """
    A single chunk returned from the vector store.
    Typed explicitly so every part of the system knows
    exactly what shape a retrieved chunk has.
    """
    id: str
    content: str
    score: float
    metadata: dict  # ticker, year, section, page, chunk_type


class Citation(TypedDict):
    """
    A source citation attached to the final answer.
    Every claim in the answer should trace back to a citation.
    This is what separates a hallucinating chatbot from a
    trustworthy financial research tool.
    """
    ticker: str
    year: int
    section: str
    page: int
    chunk_type: str


class AgentState(TypedDict):
    """
    The complete state of one agent run.
    Flows through every node in the LangGraph graph.

    Design principle: nodes should only READ what they need
    and only WRITE what they produce. No node should mutate
    fields it doesn't own.

    Field ownership:
    - query, ticker, year, quarter    → set by caller, never mutated
    - query_type                      → set by analyze_query node
    - retrieved_chunks                → set by retrieve node
    - tool_results                    → set by tool nodes
    - final_answer, citations         → set by generate node
    - error                           → set by any node on failure
    """

    # --- Input fields (set once by the caller) ---
    query: str                          # the user's question
    ticker: str                         # e.g. "AAPL"
    year: int                           # e.g. 2025
    quarter: str                        # "annual", "Q1", "Q2", "Q3"

    # --- Router fields (set by analyze_query node) ---
    query_type: Optional[str]           # "retrieval" | "comparison" | "calculation" | "summary"
    comparison_year: Optional[int]      # second year for comparison queries e.g. 2024

    # --- Retrieval fields (set by retrieve node) ---
    retrieved_chunks: Optional[list[RetrievedChunk]]
    section_filter: Optional[str]       # e.g. "MD&A" — narrows vector search

    # --- Tool output fields (set by tool nodes) ---
    tool_results: Optional[dict[str, Any]]  # flexible dict for tool outputs

    # --- Output fields (set by generate node) ---
    final_answer: Optional[str]         # the response shown to the user
    citations: Optional[list[Citation]] # sources backing the answer

    # --- Control fields ---
    error: Optional[str]                # non-None means something went wrong
    comparison_ticker: Optional[str]   # second ticker for cross-company queries


class SpecialistState(TypedDict):
    """
    State for each specialist sub-agent graph (retrieve → generate).
    Populated by the supervisor before invoking a specialist subgraph.
    """
    query: str
    ticker: str
    year: int
    quarter: str
    section_filter: Optional[str]
    comparison_year: Optional[int]
    comparison_ticker: Optional[str]
    retrieved_chunks: Optional[list[RetrievedChunk]]
    tool_results: Optional[dict[str, Any]]
    final_answer: Optional[str]
    citations: Optional[list[Citation]]
    error: Optional[str]


class SupervisorState(TypedDict):
    """
    State for the supervisor graph that classifies queries and routes to specialists.

    The supervisor owns query_type, section_filter, and comparison fields.
    Specialists write final_answer, citations, and retrieved_chunks back up.
    """
    # Input (set by caller, never mutated)
    query: str
    ticker: str
    year: int
    quarter: str
    # Routing (set by classify node)
    query_type: Optional[str]          # "retrieval" | "comparison" | "calculation" | "summary" | "cross_company"
    comparison_year: Optional[int]
    comparison_ticker: Optional[str]
    section_filter: Optional[str]
    # Output (written by specialist agents)
    retrieved_chunks: Optional[list[RetrievedChunk]]
    final_answer: Optional[str]
    citations: Optional[list[Citation]]
    error: Optional[str]
    # Guardrail (set by guardrail node)
    is_out_of_scope: Optional[bool]