# agents/tools/summarize.py

from agents.state import RetrievedChunk


# How many chunks to use per section summary
# More chunks = more complete but more expensive
SUMMARY_CHUNK_LIMIT = 8


class SummarizeSectionTool:
    """
    Prepares context for summarizing an entire SEC filing section.

    Rather than summarizing directly (which would require passing
    all chunks to the LLM in one call), this tool:
    1. Selects the most representative chunks from a section
    2. Prioritizes headers and early chunks (document structure)
    3. Always includes table chunks (they're information-dense)
    4. Packages everything into a structured prompt

    This keeps token usage predictable and avoids context window issues.
    """

    def run(
        self,
        section_name: str,
        chunks: list[RetrievedChunk],
        ticker: str,
        year: int,
    ) -> dict:
        """
        Selects and organizes chunks for section summarization.

        Returns a dict with the prepared context and summary prompt.
        """
        # Filter to the requested section
        section_chunks = [
            c for c in chunks
            if c["metadata"].get("section", "").lower() == section_name.lower()
        ]

        if not section_chunks:
            return {
                "error": f"No chunks found for section '{section_name}'",
                "section": section_name,
                "chunks_used": 0,
            }

        # Prioritization strategy:
        # 1. Tables first — dense financial information
        # 2. Then text ordered by page — preserves document flow
        tables = [c for c in section_chunks if c["metadata"].get("chunk_type") == "table"]
        texts = sorted(
            [c for c in section_chunks if c["metadata"].get("chunk_type") == "text"],
            key=lambda x: x["metadata"].get("page", 0)
        )

        # Take tables first, fill remainder with text up to limit
        selected = tables[:3] + texts[:SUMMARY_CHUNK_LIMIT - len(tables[:3])]
        context = "\n\n---\n\n".join([c["content"] for c in selected])

        summary_prompt = f"""Summarize the '{section_name}' section from {ticker}'s {year} 10-K filing.

## Source Content ({len(selected)} chunks from {len(section_chunks)} total)
{context}

## Summary Requirements
Provide a structured summary with:
1. **Key Points** (3-5 bullet points of the most important information)
2. **Notable Numbers** (specific figures, percentages, or metrics mentioned)
3. **Year-over-Year Changes** (if any comparative data is present)
4. **Risk Flags** (anything that signals concern or uncertainty)

Keep the summary concise but complete. Cite page numbers for key claims."""

        return {
            "section": section_name,
            "ticker": ticker,
            "year": year,
            "chunks_used": len(selected),
            "total_section_chunks": len(section_chunks),
            "summary_prompt": summary_prompt,
            "selected_chunks": selected,
        }