# agents/specialists/base.py
#
# Shared utilities used by all specialist agents.
# Keeping these here avoids circular imports and prevents duplication
# across the five specialist modules.

from openai import OpenAI
from dotenv import load_dotenv
from agents.state import Citation, RetrievedChunk

load_dotenv()

# Single OpenAI client shared across all specialists — thread-safe and avoids
# re-reading the API key on every agent init.
client = OpenAI()


def extract_citations(chunks: list[RetrievedChunk]) -> list[Citation]:
    """Builds a deduplicated, page-sorted citation list from retrieved chunks."""
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


def format_chunks(chunks: list[RetrievedChunk]) -> str:
    """Formats chunks into a readable context block for LLM prompts."""
    parts = []
    for i, chunk in enumerate(chunks):
        meta = chunk["metadata"]
        parts.append(
            f"[Chunk {i+1} | {meta.get('section')} | Page {meta.get('page')} | Score: {chunk['score']:.3f}]\n"
            f"{chunk['content']}"
        )
    return "\n\n".join(parts)


def specialist_input(supervisor_state: dict, **overrides) -> dict:
    """
    Builds a SpecialistState dict from SupervisorState, with optional field overrides.
    Centralises the state-mapping boilerplate so each specialist's invoke() is one line.
    """
    base = {
        "query": supervisor_state["query"],
        "ticker": supervisor_state["ticker"],
        "year": supervisor_state["year"],
        "quarter": supervisor_state["quarter"],
        "section_filter": supervisor_state.get("section_filter"),
        "comparison_year": supervisor_state.get("comparison_year"),
        "comparison_ticker": supervisor_state.get("comparison_ticker"),
        "retrieved_chunks": None,
        "tool_results": None,
        "final_answer": None,
        "citations": None,
        "error": None,
    }
    base.update(overrides)
    return base
