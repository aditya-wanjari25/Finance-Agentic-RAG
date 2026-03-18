# agents/nodes.py

import json
from openai import OpenAI
from dotenv import load_dotenv
from retrieval.vector_store import get_vector_store
from agents.state import AgentState, Citation
from agents.tools.retrieve import RetrieveTool
from agents.tools.compare import ComparePeriodsTool
from agents.tools.calculate import CalculateRatioTool
from agents.tools.summarize import SummarizeSectionTool
from agents.prompts.system import SYSTEM_PROMPT
from agents.prompts.templates import (
    QUERY_ANALYSIS_TEMPLATE,
    GENERATION_TEMPLATE,
    COMPARISON_TEMPLATE,
)

load_dotenv()

client = OpenAI()

# Initialize tools once — they're stateless so safe to share
retrieve_tool = RetrieveTool()
compare_tool = ComparePeriodsTool()
calculate_tool = CalculateRatioTool()
summarize_tool = SummarizeSectionTool()


def analyze_query(state: AgentState) -> dict:
    """
    Node 1: Classifies the query and extracts routing metadata.

    Sends the query to GPT-4o with a structured prompt asking for:
    - query_type: how should we handle this?
    - section_filter: which SEC section is most relevant?
    - comparison_year: is a second year needed?

    We ask for JSON output and parse it — this is a simple
    but effective way to get structured data from an LLM.
    """
    print(f"\n🧠 Analyzing query: '{state['query']}'")

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
            temperature=0,        # deterministic — classification should not be creative
            response_format={"type": "json_object"},
        )

        result = json.loads(response.choices[0].message.content)
        query_type = result.get("query_type", "retrieval")
        section_filter = result.get("section_filter")
        comparison_year = result.get("comparison_year")

        print(f"  → Query type: {query_type}")
        print(f"  → Section filter: {section_filter}")
        print(f"  → Comparison year: {comparison_year}")

        return {
            "query_type": query_type,
            "section_filter": section_filter,
            "comparison_year": comparison_year,
        }

    except Exception as e:
        print(f"  ⚠️  Query analysis failed: {e}. Defaulting to retrieval.")
        return {
            "query_type": "retrieval",
            "section_filter": None,
            "comparison_year": None,
            "error": str(e),
        }

def retrieve(state: AgentState) -> dict:
    """
    Node 2: Retrieves relevant chunks based on query type.

    Routes to different retrieval strategies:
    - comparison → ComparePeriodsTool (two-year retrieval)
    - summary    → retrieve all chunks from a section
    - calculation → retrieve with preference for tables
    - retrieval  → standard filtered semantic search
    """
    print(f"\n🔍 Retrieving context (type: {state['query_type']})...")

    query_type = state.get("query_type", "retrieval")
    ticker = state["ticker"]
    year = state["year"]
    query = state["query"]
    section_filter = state.get("section_filter")

    try:
        if query_type == "comparison" and state.get("comparison_year"):
            # Two-period retrieval
            result = compare_tool.run(
                query=query,
                ticker=ticker,
                year_current=year,
                year_comparison=state["comparison_year"],
                section_filter=section_filter,
                n_results=4,
            )
            return {
                "retrieved_chunks": result["current"],
                "tool_results": {"comparison": result},
            }

        elif query_type == "summary":
            store = get_vector_store()

            # Use keyword args — both ChromaDB and OpenSearch wrappers accept these
            all_section_chunks = store.get_by_metadata(
                ticker=ticker,
                year=year,
                section=section_filter or "MD&A",
                limit=50,
            )
            summary_result = summarize_tool.run(
                section_name=section_filter or "MD&A",
                chunks=all_section_chunks,
                ticker=ticker,
                year=year,
            )
            return {
                "retrieved_chunks": all_section_chunks,
                "tool_results": {"summary": summary_result},
            }

        elif query_type == "calculation":
            # Prefer table chunks for calculations
            chunks = retrieve_tool.run(
                query=query,
                ticker=ticker,
                year=year,
                n_results=6,
                section_filter=section_filter,
                chunk_type_filter=None,  # get both tables and text
            )
            calc_result = calculate_tool.run(
                ratio_name=_detect_ratio(query),
                chunks=chunks,
            )
            return {
                "retrieved_chunks": chunks,
                "tool_results": {"calculation": calc_result},
            }

        else:
            # Standard retrieval
            chunks = retrieve_tool.run(
                query=query,
                ticker=ticker,
                year=year,
                n_results=5,
                section_filter=section_filter,
            )
            return {"retrieved_chunks": chunks}

    except Exception as e:
        print(f"  ⚠️  Retrieval failed: {e}")
        return {
            "retrieved_chunks": [],
            "error": str(e),
        }


def _detect_ratio(query: str) -> str:
    """
    Simple keyword matching to identify which ratio is being asked about.
    Maps common phrasings to our standardized ratio names.
    """
    query_lower = query.lower()
    if any(w in query_lower for w in ["gross margin", "gross profit margin"]):
        return "gross_margin"
    elif any(w in query_lower for w in ["operating margin", "operating income margin"]):
        return "operating_margin"
    elif any(w in query_lower for w in ["net margin", "net income margin", "profit margin"]):
        return "net_margin"
    elif any(w in query_lower for w in ["revenue growth", "sales growth"]):
        return "revenue_growth"
    elif any(w in query_lower for w in ["debt to equity", "leverage"]):
        return "debt_to_equity"
    elif any(w in query_lower for w in ["current ratio", "liquidity"]):
        return "current_ratio"
    else:
        return "gross_margin"  # sensible default for finance queries

def generate_answer(state: AgentState) -> dict:
    """
    Node 3: Generates the final answer using retrieved context.

    Selects the right prompt template based on query type,
    formats the context, calls GPT-4o, and extracts citations
    from the retrieved chunks used.
    """
    print(f"\n✍️  Generating answer...")

    query_type = state.get("query_type", "retrieval")
    chunks = state.get("retrieved_chunks", [])
    tool_results = state.get("tool_results", {})

    try:
        if query_type == "comparison" and "comparison" in (tool_results or {}):
            messages = _build_comparison_messages(state)
        else:
            messages = _build_standard_messages(state)

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.1,    # slight creativity for natural language, but mostly factual
        )

        answer = response.choices[0].message.content

        # Extract citations from the chunks we actually used
        citations = _extract_citations(chunks)

        print(f"  ✅ Answer generated ({len(answer)} chars, {len(citations)} citations)")

        return {
            "final_answer": answer,
            "citations": citations,
        }

    except Exception as e:
        print(f"  ⚠️  Generation failed: {e}")
        return {
            "final_answer": f"I encountered an error generating the answer: {e}",
            "citations": [],
            "error": str(e),
        }


def _build_standard_messages(state: AgentState) -> list:
    """Builds the message list for standard retrieval and calculation queries."""
    chunks = state.get("retrieved_chunks", [])
    tool_results = state.get("tool_results", {})

    # Format chunks into readable context block
    context_parts = []
    for i, chunk in enumerate(chunks):
        meta = chunk["metadata"]
        context_parts.append(
            f"[Chunk {i+1} | {meta.get('section')} | Page {meta.get('page')} | Score: {chunk['score']:.3f}]\n"
            f"{chunk['content']}"
        )
    context = "\n\n".join(context_parts)

    # Format tool results if present
    tool_str = "None"
    if tool_results:
        if "calculation" in tool_results:
            tool_str = tool_results["calculation"].get("extraction_prompt", "")
        elif "summary" in tool_results:
            tool_str = tool_results["summary"].get("summary_prompt", "")

    user_message = GENERATION_TEMPLATE.format(
        query=state["query"],
        ticker=state["ticker"],
        year=state["year"],
        quarter=state["quarter"],
        context=context,
        tool_results=tool_str,
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]


def _build_comparison_messages(state: AgentState) -> list:
    """Builds the message list for comparison queries."""
    comparison_data = state["tool_results"]["comparison"]

    def format_chunks(chunks):
        parts = []
        for i, chunk in enumerate(chunks):
            meta = chunk["metadata"]
            parts.append(
                f"[Chunk {i+1} | {meta.get('section')} | Page {meta.get('page')}]\n"
                f"{chunk['content']}"
            )
        return "\n\n".join(parts)

    user_message = COMPARISON_TEMPLATE.format(
        query=state["query"],
        year=comparison_data["year_current"],
        comparison_year=comparison_data["year_comparison"],
        context_current=format_chunks(comparison_data["current"]),
        context_comparison=format_chunks(comparison_data.get("comparison", [])),
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]


def _extract_citations(chunks: list) -> list[Citation]:
    """Builds citation list from the chunks used in the answer."""
    citations = []
    seen = set()

    for chunk in chunks:
        meta = chunk["metadata"]
        key = (meta.get("section"), meta.get("page"))
        if key not in seen:
            seen.add(key)
            citations.append({
                "ticker": meta.get("ticker", ""),
                "year": meta.get("year", 0),
                "section": meta.get("section", ""),
                "page": meta.get("page", 0),
                "chunk_type": meta.get("chunk_type", ""),
            })

    return sorted(citations, key=lambda x: x["page"])

def handle_error(state: AgentState) -> dict:
    """
    Node 4: Graceful error handling.
    Returns a user-friendly message instead of crashing.
    """
    error = state.get("error", "Unknown error")
    print(f"\n❌ Error handler triggered: {error}")
    return {
        "final_answer": (
            f"I was unable to complete your request due to an error: {error}\n\n"
            f"Please verify that the document for {state['ticker']} {state['year']} "
            f"has been ingested and try again."
        ),
        "citations": [],
    }


def route_after_analysis(state: AgentState) -> str:
    """
    Conditional edge function — called by LangGraph after analyze_query.
    Returns the name of the next node to visit.
    If there was an error in analysis, go straight to error handler.
    """
    if state.get("error"):
        return "handle_error"
    return "retrieve"