# agents/tools/calculate.py

import re
from agents.state import RetrievedChunk


# Common financial ratios the agent can calculate
# Maps ratio name → formula description for the LLM
RATIO_FORMULAS = {
    "gross_margin": "gross_profit / revenue * 100",
    "operating_margin": "operating_income / revenue * 100",
    "net_margin": "net_income / revenue * 100",
    "revenue_growth": "(revenue_current - revenue_prior) / revenue_prior * 100",
    "debt_to_equity": "total_debt / shareholders_equity",
    "current_ratio": "current_assets / current_liabilities",
    "eps_growth": "(eps_current - eps_prior) / abs(eps_prior) * 100",
}


class CalculateRatioTool:
    """
    Extracts financial figures from retrieved chunks and calculates ratios.

    Design decision: we don't parse numbers ourselves (too brittle for
    varied PDF formats). Instead we pass the relevant chunks to the LLM
    with explicit instructions to extract and calculate, then verify
    the result is a reasonable number.

    This hybrid approach is more robust than pure regex extraction
    while being more verifiable than asking the LLM to recall figures
    from memory.
    """

    def run(
        self,
        ratio_name: str,
        chunks: list[RetrievedChunk],
    ) -> dict:
        """
        Prepares calculation context from chunks.

        Returns a dict with:
        - ratio_name: what we're calculating
        - formula: the formula to apply
        - relevant_chunks: filtered chunks most likely to contain the numbers
        - extraction_prompt: ready-to-use prompt for the LLM to extract + calculate
        """
        formula = RATIO_FORMULAS.get(ratio_name, f"Calculate {ratio_name}")

        # Filter to table chunks first — they're most likely to have exact figures
        table_chunks = [c for c in chunks if c["metadata"].get("chunk_type") == "table"]
        financial_chunks = [c for c in chunks if c["metadata"].get("section") == "Financial Statements"]

        # Prefer tables, fall back to financial statement text
        relevant = table_chunks if table_chunks else financial_chunks if financial_chunks else chunks

        context = "\n\n".join([c["content"] for c in relevant[:3]])

        extraction_prompt = f"""From the following financial data, extract the numbers needed and calculate: {formula}

Financial Data:
{context}

Steps:
1. Identify the relevant figures with their exact values and units
2. Apply the formula: {formula}
3. State the result clearly with units (% for margins, x for ratios)
4. Cite which table or section the numbers came from

If the required figures are not present in the data, explicitly state what is missing."""

        return {
            "ratio_name": ratio_name,
            "formula": formula,
            "relevant_chunks": relevant[:3],
            "extraction_prompt": extraction_prompt,
        }

    def list_available_ratios(self) -> list[str]:
        return list(RATIO_FORMULAS.keys())