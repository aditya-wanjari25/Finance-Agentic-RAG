# agents/graph.py

from langgraph.graph import StateGraph, END
from agents.state import AgentState
from agents.nodes import (
    analyze_query,
    retrieve,
    generate_answer,
    handle_error,
    route_after_analysis,
)
from agents.observability import get_langsmith_config, is_tracing_enabled


def build_graph() -> StateGraph:
    """
    Assembles the LangGraph agent.

    Graph flow:
    START → analyze_query → [route] → retrieve → generate_answer → END
                                ↓
                           handle_error → END
    """
    graph = StateGraph(AgentState)

    graph.add_node("analyze_query", analyze_query)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate_answer", generate_answer)
    graph.add_node("handle_error", handle_error)

    graph.set_entry_point("analyze_query")

    graph.add_conditional_edges(
        "analyze_query",
        route_after_analysis,
        {
            "retrieve": "retrieve",
            "handle_error": "handle_error",
        }
    )

    graph.add_edge("retrieve", "generate_answer")
    graph.add_edge("generate_answer", END)
    graph.add_edge("handle_error", END)

    return graph.compile()


agent = build_graph()


def run_query(
    query: str,
    ticker: str,
    year: int,
    quarter: str = "annual",
) -> dict:
    """
    Clean public interface for running the agent.
    Automatically attaches LangSmith tracing metadata when enabled.
    """
    initial_state: AgentState = {
        "query": query,
        "ticker": ticker,
        "year": year,
        "quarter": quarter,
        "query_type": None,
        "comparison_year": None,
        "comparison_ticker": None,
        "retrieved_chunks": None,
        "section_filter": None,
        "tool_results": None,
        "final_answer": None,
        "citations": None,
        "error": None,
    }

    # Build LangSmith config with rich metadata
    # This metadata appears in the LangSmith UI for filtering and debugging
    langsmith_config = get_langsmith_config(
        run_name=f"{ticker} {year} — {query[:50]}",
        tags=[ticker, str(year), quarter],
        metadata={
            "ticker": ticker,
            "year": year,
            "quarter": quarter,
            "query_preview": query[:100],
        }
    )

    if langsmith_config:
        final_state = agent.invoke(initial_state, config=langsmith_config)
    else:
        final_state = agent.invoke(initial_state)

    return final_state