# agents/specialists/cross_company_agent.py
#
# Handles cross-company comparison queries:
#   "Compare Apple and Google's risk factors"
#   "How does Microsoft's revenue compare to Amazon's?"

from langgraph.graph import StateGraph, END
from agents.state import SpecialistState, SupervisorState
from agents.tools.retrieve import RetrieveTool
from agents.prompts.system import SYSTEM_PROMPT
from agents.prompts.templates import CROSS_COMPANY_TEMPLATE
from agents.specialists.base import client, extract_citations, specialist_input


def _format_company_chunks(chunks: list) -> str:
    parts = []
    for i, chunk in enumerate(chunks):
        meta = chunk["metadata"]
        parts.append(
            f"[{meta.get('section')} | Page {meta.get('page')}]\n"
            f"{chunk['content']}"
        )
    return "\n\n".join(parts)


class CrossCompanyAgent:
    def __init__(self):
        self.retrieve_tool = RetrieveTool()
        self.graph = self._build_graph()

    # -----------------------------------------------------------------
    # Internal graph nodes
    # -----------------------------------------------------------------

    def _retrieve(self, state: SpecialistState) -> dict:
        ticker1 = state["ticker"]
        ticker2 = state["comparison_ticker"]
        print(f"  [CrossCompanyAgent] Retrieving {ticker1} and {ticker2} context...")

        # Sequential retrieval — both companies, same query, same year
        chunks_1 = self.retrieve_tool.run(
            query=state["query"],
            ticker=ticker1,
            year=state["year"],
            n_results=5,
            section_filter=state.get("section_filter"),
        )
        chunks_2 = self.retrieve_tool.run(
            query=state["query"],
            ticker=ticker2,
            year=state["year"],
            n_results=5,
            section_filter=state.get("section_filter"),
        )
        print(f"  [CrossCompanyAgent] {len(chunks_1)} chunks for {ticker1}, "
              f"{len(chunks_2)} chunks for {ticker2}")
        return {
            "retrieved_chunks": chunks_1 + chunks_2,
            "tool_results": {
                "cross_company": {
                    "ticker1": ticker1,
                    "ticker2": ticker2,
                    "chunks_ticker1": chunks_1,
                    "chunks_ticker2": chunks_2,
                }
            },
        }

    def _generate(self, state: SpecialistState) -> dict:
        print(f"  [CrossCompanyAgent] Generating cross-company comparison...")
        cross_data = state["tool_results"]["cross_company"]

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": CROSS_COMPANY_TEMPLATE.format(
                    query=state["query"],
                    ticker=cross_data["ticker1"],
                    comparison_ticker=cross_data["ticker2"],
                    year=state["year"],
                    context_ticker1=_format_company_chunks(cross_data["chunks_ticker1"]),
                    context_ticker2=_format_company_chunks(cross_data["chunks_ticker2"]),
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
