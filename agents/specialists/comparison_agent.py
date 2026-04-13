# agents/specialists/comparison_agent.py
#
# Handles cross-period comparison queries:
#   "How did gross margin change from 2024 to 2025?"
#   "Compare iPhone revenue across years"

from langgraph.graph import StateGraph, END
from agents.state import SpecialistState, SupervisorState
from agents.tools.compare import ComparePeriodsTool
from agents.prompts.system import SYSTEM_PROMPT
from agents.prompts.templates import COMPARISON_TEMPLATE
from agents.specialists.base import client, extract_citations, specialist_input


def _format_period_chunks(chunks: list) -> str:
    parts = []
    for i, chunk in enumerate(chunks):
        meta = chunk["metadata"]
        parts.append(
            f"[Chunk {i+1} | {meta.get('section')} | Page {meta.get('page')}]\n"
            f"{chunk['content']}"
        )
    return "\n\n".join(parts)


class ComparisonAgent:
    def __init__(self):
        self.compare_tool = ComparePeriodsTool()
        self.graph = self._build_graph()

    # -----------------------------------------------------------------
    # Internal graph nodes
    # -----------------------------------------------------------------

    def _retrieve(self, state: SpecialistState) -> dict:
        print(f"  [ComparisonAgent] Two-period retrieval — {state['year']} vs {state['comparison_year']}")
        result = self.compare_tool.run(
            query=state["query"],
            ticker=state["ticker"],
            year_current=state["year"],
            year_comparison=state["comparison_year"],
            section_filter=state.get("section_filter"),
            n_results=4,
        )
        all_chunks = result["current"] + result.get("comparison", [])
        print(f"  [ComparisonAgent] Retrieved {len(all_chunks)} chunks across both periods")
        return {
            "retrieved_chunks": all_chunks,
            "tool_results": {"comparison": result},
        }

    def _generate(self, state: SpecialistState) -> dict:
        print(f"  [ComparisonAgent] Generating comparison answer...")
        data = state["tool_results"]["comparison"]

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": COMPARISON_TEMPLATE.format(
                    query=state["query"],
                    year=data["year_current"],
                    comparison_year=data["year_comparison"],
                    context_current=_format_period_chunks(data["current"]),
                    context_comparison=_format_period_chunks(data.get("comparison", [])),
                )},
            ],
            temperature=0.1,
        )

        return {
            "final_answer": response.choices[0].message.content,
            "citations": extract_citations(state.get("retrieved_chunks", [])),
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
