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


def build_graph() -> StateGraph:
    """
    Assembles the LangGraph agent.

    Graph flow:
    START → analyze_query → [route] → retrieve → generate_answer → END
                                ↓
                           handle_error → END
    """
    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("analyze_query", analyze_query)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate_answer", generate_answer)
    graph.add_node("handle_error", handle_error)

    # Entry point
    graph.set_entry_point("analyze_query")

    # Conditional edge after analysis
    # If error → handle_error, else → retrieve
    graph.add_conditional_edges(
        "analyze_query",
        route_after_analysis,
        {
            "retrieve": "retrieve",
            "handle_error": "handle_error",
        }
    )

    # Linear edges for the happy path
    graph.add_edge("retrieve", "generate_answer")
    graph.add_edge("generate_answer", END)
    graph.add_edge("handle_error", END)

    return graph.compile()


# Compiled graph — import this everywhere
agent = build_graph()


def run_query(
    query: str,
    ticker: str,
    year: int,
    quarter: str = "annual",
) -> dict:
    """
    Clean public interface for running the agent.
    Returns the final state after the graph completes.
    """
    initial_state: AgentState = {
        "query": query,
        "ticker": ticker,
        "year": year,
        "quarter": quarter,
        "query_type": None,
        "comparison_year": None,
        "retrieved_chunks": None,
        "section_filter": None,
        "tool_results": None,
        "final_answer": None,
        "citations": None,
        "error": None,
    }

    final_state = agent.invoke(initial_state)
    return final_state