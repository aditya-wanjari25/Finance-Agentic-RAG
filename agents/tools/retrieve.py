# agents/tools/retrieve.py

from retrieval.hybrid_retriever import HybridRetriever
from agents.state import RetrievedChunk
from dotenv import load_dotenv

load_dotenv()


class RetrieveTool:
    """
    Hybrid retrieval tool with metadata filtering.

    Combines BM25 (keyword) and vector (semantic) search via Reciprocal Rank Fusion.

    The key difference from naive RAG:
    - Naive RAG: embed query → search ALL chunks → return top 5
    - Our approach: filter by ticker/year/section first, then run hybrid search within that subset

    Why does this matter for finance?
    If you have 5 companies × 3 years = 15 documents in your vector store,
    a naive search for "revenue risk" might return chunks from the wrong
    company or wrong year. Metadata filtering prevents this entirely.
    """

    def __init__(self, collection_name: str = "finsight"):
        self.retriever = HybridRetriever()

    def run(
        self,
        query: str,
        ticker: str,
        year: int,
        n_results: int = 5,
        section_filter: str = None,
        chunk_type_filter: str = None,  # "text" | "table" | None
    ) -> list[RetrievedChunk]:
        """
        Runs hybrid BM25 + vector retrieval and returns the top-n chunks.

        Args:
            query:              The search query (usually the user's question)
            ticker:             Filter to this company only
            year:               Filter to this fiscal year only
            n_results:          How many chunks to return
            section_filter:     Optional — narrow to a specific SEC section
            chunk_type_filter:  Optional — return only tables or only text

        Returns:
            List of RetrievedChunk dicts ordered by RRF score
        """
        return self.retriever.retrieve(
            query=query,
            ticker=ticker,
            year=year,
            n_results=n_results,
            section_filter=section_filter,
            chunk_type_filter=chunk_type_filter,
        )

