# agents/specialists/summarization_agent.py
#
# Handles full-section summarization queries:
#   "Summarize the MD&A section"
#   "Give me the key points from Risk Factors"

from langgraph.graph import StateGraph, END
from agents.state import SpecialistState, SupervisorState
from agents.tools.summarize import SummarizeSectionTool
from agents.prompts.system import SYSTEM_PROMPT
from agents.prompts.templates import GENERATION_TEMPLATE
from agents.specialists.base import client, extract_citations, format_chunks, specialist_input
from retrieval.vector_store import get_vector_store


class SummarizationAgent:
    def __init__(self):
        self.summarize_tool = SummarizeSectionTool()
        self.graph = self._build_graph()

    # -----------------------------------------------------------------
    # Internal graph nodes
    # -----------------------------------------------------------------

    def _retrieve(self, state: SpecialistState) -> dict:
        section = state.get("section_filter") or "MD&A"
        print(f"  [SummarizationAgent] Fetching all chunks for section: {section}")

        # Use get_by_metadata instead of semantic search — we want EVERY chunk
        # from this section ordered by page, not the top-5 by relevance
        store = get_vector_store()
        all_chunks = store.get_by_metadata(
            ticker=state["ticker"],
            year=state["year"],
            section=section,
            limit=50,
        )
        summary_result = self.summarize_tool.run(
            section_name=section,
            chunks=all_chunks,
            ticker=state["ticker"],
            year=state["year"],
        )
        print(f"  [SummarizationAgent] Retrieved {len(all_chunks)} chunks, "
              f"using {summary_result.get('chunks_used', 0)} for summary")
        return {
            "retrieved_chunks": all_chunks,
            "tool_results": {"summary": summary_result},
        }

    def _generate(self, state: SpecialistState) -> dict:
        print(f"  [SummarizationAgent] Generating section summary...")
        chunks = state.get("retrieved_chunks", [])
        summary_result = state.get("tool_results", {}).get("summary", {})

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
                    tool_results=summary_result.get("summary_prompt", "None"),
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
