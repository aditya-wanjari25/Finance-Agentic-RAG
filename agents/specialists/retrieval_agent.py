# agents/specialists/retrieval_agent.py
#
# Handles standard semantic retrieval queries:
#   "What are Apple's main risk factors?"
#   "What did the company say about China revenue?"

from langgraph.graph import StateGraph, END
from agents.state import SpecialistState, SupervisorState
from agents.tools.retrieve import RetrieveTool
from agents.prompts.system import SYSTEM_PROMPT
from agents.prompts.templates import GENERATION_TEMPLATE
from agents.specialists.base import client, extract_citations, format_chunks, specialist_input


class RetrievalAgent:
    def __init__(self):
        self.retrieve_tool = RetrieveTool()
        self.graph = self._build_graph()

    # -----------------------------------------------------------------
    # Internal graph nodes
    # -----------------------------------------------------------------

    def _retrieve(self, state: SpecialistState) -> dict:
        print(f"  [RetrievalAgent] Semantic search — ticker={state['ticker']} year={state['year']}")
        chunks = self.retrieve_tool.run(
            query=state["query"],
            ticker=state["ticker"],
            year=state["year"],
            n_results=5,
            section_filter=state.get("section_filter"),
        )
        print(f"  [RetrievalAgent] Retrieved {len(chunks)} chunks")
        return {"retrieved_chunks": chunks}

    def _generate(self, state: SpecialistState) -> dict:
        print(f"  [RetrievalAgent] Generating answer...")
        chunks = state.get("retrieved_chunks", [])

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
                    tool_results="None",
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
