# agents/specialists/calculation_agent.py
#
# Handles financial ratio calculation queries:
#   "What is Apple's gross margin for 2025?"
#   "What's the debt-to-equity ratio?"

from langgraph.graph import StateGraph, END
from agents.state import SpecialistState, SupervisorState
from agents.tools.retrieve import RetrieveTool
from agents.tools.calculate import CalculateRatioTool
from agents.prompts.system import SYSTEM_PROMPT
from agents.prompts.templates import GENERATION_TEMPLATE
from agents.specialists.base import client, extract_citations, format_chunks, specialist_input


def _detect_ratio(query: str) -> str:
    """Maps common query phrasings to standardized ratio names."""
    q = query.lower()
    if any(w in q for w in ["gross margin", "gross profit margin"]):
        return "gross_margin"
    if any(w in q for w in ["operating margin", "operating income margin"]):
        return "operating_margin"
    if any(w in q for w in ["net margin", "net income margin", "profit margin"]):
        return "net_margin"
    if any(w in q for w in ["revenue growth", "sales growth"]):
        return "revenue_growth"
    if any(w in q for w in ["debt to equity", "leverage"]):
        return "debt_to_equity"
    if any(w in q for w in ["current ratio", "liquidity"]):
        return "current_ratio"
    return "gross_margin"


class CalculationAgent:
    def __init__(self):
        self.retrieve_tool = RetrieveTool()
        self.calculate_tool = CalculateRatioTool()
        self.graph = self._build_graph()

    # -----------------------------------------------------------------
    # Internal graph nodes
    # -----------------------------------------------------------------

    def _retrieve(self, state: SpecialistState) -> dict:
        ratio = _detect_ratio(state["query"])
        print(f"  [CalculationAgent] Retrieving financial data for ratio: {ratio}")

        # Fetch more chunks than other agents — calculations need table chunks
        # which may not score highest in semantic search
        chunks = self.retrieve_tool.run(
            query=state["query"],
            ticker=state["ticker"],
            year=state["year"],
            n_results=6,
            section_filter=state.get("section_filter"),
        )
        calc_result = self.calculate_tool.run(ratio_name=ratio, chunks=chunks)
        print(f"  [CalculationAgent] Retrieved {len(chunks)} chunks, "
              f"{len(calc_result['relevant_chunks'])} relevant to calculation")
        return {
            "retrieved_chunks": chunks,
            "tool_results": {"calculation": calc_result},
        }

    def _generate(self, state: SpecialistState) -> dict:
        print(f"  [CalculationAgent] Generating calculation answer...")
        chunks = state.get("retrieved_chunks", [])
        calc_result = state.get("tool_results", {}).get("calculation", {})

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": GENERATION_TEMPLATE.format(
                    query=state["query"],
                    ticker=state["ticker"],
                    year=state["year"],
                    quarter=state["quarter"],
                    context=format_chunks(chunks),
                    tool_results=calc_result.get("extraction_prompt", "None"),
                )},
            ],
            temperature=0.1,
        )

        return {
            "final_answer": response.choices[0].message.content,
            "citations": extract_citations(chunks),
        }

    def _build_graph(self):
        g = StateGraph(SpecialistState)
        g.add_node("retrieve", self._retrieve)
        g.add_node("generate", self._generate)
        g.set_entry_point("retrieve")
        g.add_edge("retrieve", "generate")
        g.add_edge("generate", END)
        return g.compile()

    # -----------------------------------------------------------------
    # Public interface called by the supervisor
    # -----------------------------------------------------------------

    def invoke(self, state: SupervisorState) -> dict:
        result = self.graph.invoke(specialist_input(state))
        return {
            "final_answer": result["final_answer"],
            "citations": result["citations"],
            "retrieved_chunks": result["retrieved_chunks"],
        }
