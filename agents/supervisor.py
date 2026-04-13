# agents/supervisor.py
#
# The supervisor graph: classifies incoming queries and routes them to the
# appropriate specialist agent. Each specialist is an independent LangGraph
# subgraph with its own retrieve → generate pipeline.
#
# Graph shape:
#   START → classify → [route] → {specialist} → END
#                             ↘ handle_error → END

import json
from openai import OpenAI
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

from agents.state import SupervisorState
from agents.prompts.templates import QUERY_ANALYSIS_TEMPLATE
from agents.specialists.retrieval_agent import RetrievalAgent
from agents.specialists.comparison_agent import ComparisonAgent
from agents.specialists.calculation_agent import CalculationAgent
from agents.specialists.summarization_agent import SummarizationAgent
from agents.specialists.cross_company_agent import CrossCompanyAgent

load_dotenv()

client = OpenAI()

# Company name → ticker for cross-company query detection.
# Add entries here when new companies are ingested.
TICKER_MAP = {
    "apple": "AAPL", "aapl": "AAPL",
    "google": "GOOGL", "googl": "GOOGL", "alphabet": "GOOGL",
    "microsoft": "MSFT", "msft": "MSFT",
    "amazon": "AMZN", "amzn": "AMZN",
    "meta": "META", "facebook": "META",
    "nvidia": "NVDA", "nvda": "NVDA",
}

# Specialist agents — initialized once at module load (they're stateless).
_retrieval_agent = RetrievalAgent()
_comparison_agent = ComparisonAgent()
_calculation_agent = CalculationAgent()
_summarization_agent = SummarizationAgent()
_cross_company_agent = CrossCompanyAgent()


# -----------------------------------------------------------------
# Supervisor nodes
# -----------------------------------------------------------------

def classify(state: SupervisorState) -> dict:
    """
    Node 1: Classifies the query and sets routing metadata.

    Uses GPT-4o to determine query_type, section_filter, and comparison_year.
    Cross-company detection runs first via keyword matching so it can override
    the LLM classification (the LLM doesn't know which tickers are ingested).
    """
    print(f"\n🧠 [Supervisor] Classifying: '{state['query']}'")

    # Detect second ticker before calling the LLM
    query_lower = state["query"].lower()
    comparison_ticker = None
    for name, ticker in TICKER_MAP.items():
        if name in query_lower and ticker != state["ticker"]:
            comparison_ticker = ticker
            break

    prompt = QUERY_ANALYSIS_TEMPLATE.format(
        query=state["query"],
        ticker=state["ticker"],
        year=state["year"],
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a financial query classifier. Respond only in valid JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )

        result = json.loads(response.choices[0].message.content)
        query_type = result.get("query_type", "retrieval")

        # Cross-company detection overrides LLM classification
        if comparison_ticker:
            query_type = "cross_company"

        print(f"  → type={query_type} | section={result.get('section_filter')} "
              f"| comparison_year={result.get('comparison_year')} "
              f"| comparison_ticker={comparison_ticker}")

        return {
            "query_type": query_type,
            "section_filter": result.get("section_filter"),
            "comparison_year": result.get("comparison_year"),
            "comparison_ticker": comparison_ticker,
        }

    except Exception as e:
        print(f"  ⚠️  Classification failed: {e}. Defaulting to retrieval.")
        return {
            "query_type": "retrieval",
            "section_filter": None,
            "comparison_year": None,
            "comparison_ticker": None,
            "error": str(e),
        }


def handle_error(state: SupervisorState) -> dict:
    error = state.get("error", "Unknown error")
    print(f"\n❌ [Supervisor] Error handler: {error}")
    return {
        "final_answer": (
            f"Unable to complete your request: {error}\n\n"
            f"Ensure {state['ticker']} {state['year']} has been ingested and try again."
        ),
        "citations": [],
    }


# Thin wrappers so each specialist call appears as a named node in LangSmith traces
def _run_retrieval_agent(state: SupervisorState) -> dict:
    print(f"\n🤖 [Supervisor] → RetrievalAgent")
    return _retrieval_agent.invoke(state)


def _run_comparison_agent(state: SupervisorState) -> dict:
    print(f"\n🤖 [Supervisor] → ComparisonAgent")
    return _comparison_agent.invoke(state)


def _run_calculation_agent(state: SupervisorState) -> dict:
    print(f"\n🤖 [Supervisor] → CalculationAgent")
    return _calculation_agent.invoke(state)


def _run_summarization_agent(state: SupervisorState) -> dict:
    print(f"\n🤖 [Supervisor] → SummarizationAgent")
    return _summarization_agent.invoke(state)


def _run_cross_company_agent(state: SupervisorState) -> dict:
    print(f"\n🤖 [Supervisor] → CrossCompanyAgent")
    return _cross_company_agent.invoke(state)


# -----------------------------------------------------------------
# Routing
# -----------------------------------------------------------------

def _route(state: SupervisorState) -> str:
    """Maps query_type to the specialist node name."""
    if state.get("error"):
        return "handle_error"
    return {
        "retrieval":    "retrieval_agent",
        "comparison":   "comparison_agent",
        "calculation":  "calculation_agent",
        "summary":      "summarization_agent",
        "cross_company": "cross_company_agent",
    }.get(state.get("query_type", "retrieval"), "retrieval_agent")


# -----------------------------------------------------------------
# Graph assembly
# -----------------------------------------------------------------

def build_supervisor() -> StateGraph:
    graph = StateGraph(SupervisorState)

    graph.add_node("classify", classify)
    graph.add_node("retrieval_agent", _run_retrieval_agent)
    graph.add_node("comparison_agent", _run_comparison_agent)
    graph.add_node("calculation_agent", _run_calculation_agent)
    graph.add_node("summarization_agent", _run_summarization_agent)
    graph.add_node("cross_company_agent", _run_cross_company_agent)
    graph.add_node("handle_error", handle_error)

    graph.set_entry_point("classify")

    graph.add_conditional_edges(
        "classify",
        _route,
        {
            "retrieval_agent":    "retrieval_agent",
            "comparison_agent":   "comparison_agent",
            "calculation_agent":  "calculation_agent",
            "summarization_agent": "summarization_agent",
            "cross_company_agent": "cross_company_agent",
            "handle_error":       "handle_error",
        },
    )

    for node in [
        "retrieval_agent", "comparison_agent", "calculation_agent",
        "summarization_agent", "cross_company_agent", "handle_error",
    ]:
        graph.add_edge(node, END)

    return graph.compile()
