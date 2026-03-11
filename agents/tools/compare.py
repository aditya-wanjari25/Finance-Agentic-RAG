
from agents.tools.retrieve import RetrieveTool
from agents.state import RetrievedChunk


class ComparePeriodsTool:
    """
    Compares financial data across two time periods.

    This tool handles the retrieval for both periods and packages
    the results so the generation node can write a structured comparison.

    Interview talking point: this is where multi-document RAG gets interesting.
    You need the same semantic query run against two different metadata filters,
    then the LLM synthesizes the delta.
    """

    def __init__(self, collection_name: str = "finsight"):
        self.retrieve = RetrieveTool(collection_name=collection_name)

    def run(
        self,
        query: str,
        ticker: str,
        year_current: int,
        year_comparison: int,
        section_filter: str = None,
        n_results: int = 4,
    ) -> dict:
        """
        Retrieves context for both years and returns a structured comparison dict.

        Returns:
            {
                "current":    [chunks from year_current],
                "comparison": [chunks from year_comparison],
                "year_current": 2025,
                "year_comparison": 2024,
            }
        """
        print(f"  🔍 Retrieving {ticker} {year_current} context...")
        current_chunks = self.retrieve.run(
            query=query,
            ticker=ticker,
            year=year_current,
            n_results=n_results,
            section_filter=section_filter,
        )

        print(f"  🔍 Retrieving {ticker} {year_comparison} context...")
        comparison_chunks = self.retrieve.run(
            query=query,
            ticker=ticker,
            year=year_comparison,
            n_results=n_results,
            section_filter=section_filter,
        )

        return {
            "current": current_chunks,
            "comparison": comparison_chunks,
            "year_current": year_current,
            "year_comparison": year_comparison,
            "ticker": ticker,
        }