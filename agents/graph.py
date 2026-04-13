# agents/graph.py
#
# Public entry point for the multi-agent system.
# The API and evaluation modules call run_query() — nothing else should change.
#
# Architecture:
#   SupervisorGraph (classify → route)
#     ├── RetrievalAgent    (retrieve → generate)
#     ├── ComparisonAgent   (retrieve → generate)
#     ├── CalculationAgent  (retrieve → generate)
#     ├── SummarizationAgent(retrieve → generate)
#     └── CrossCompanyAgent (retrieve → generate)

from agents.supervisor import build_supervisor
from agents.state import SupervisorState
from agents.observability import get_langsmith_config

supervisor = build_supervisor()


def run_query(
    query: str,
    ticker: str,
    year: int,
    quarter: str = "annual",
) -> dict:
    """
    Runs the multi-agent system against ingested documents.

    The supervisor classifies the query and delegates to the appropriate
    specialist agent. Each specialist has its own retrieve → generate graph.

    Returns the final SupervisorState dict, which includes:
      - final_answer (str)
      - citations    (list[Citation])
      - query_type   (str)
      - retrieved_chunks (list[RetrievedChunk])
    """
    initial_state: SupervisorState = {
        "query": query,
        "ticker": ticker,
        "year": year,
        "quarter": quarter,
        "query_type": None,
        "comparison_year": None,
        "comparison_ticker": None,
        "section_filter": None,
        "retrieved_chunks": None,
        "final_answer": None,
        "citations": None,
        "error": None,
    }

    langsmith_config = get_langsmith_config(
        run_name=f"{ticker} {year} — {query[:50]}",
        tags=[ticker, str(year), quarter],
        metadata={
            "ticker": ticker,
            "year": year,
            "quarter": quarter,
            "query_preview": query[:100],
        },
    )

    if langsmith_config:
        return supervisor.invoke(initial_state, config=langsmith_config)
    return supervisor.invoke(initial_state)
