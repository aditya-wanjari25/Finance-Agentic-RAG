# agents/tools/retrieve.py

from openai import OpenAI
from retrieval.vector_store import VectorStore
from agents.state import RetrievedChunk
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_MODEL = "text-embedding-3-small"


class RetrieveTool:
    """
    Semantic retrieval tool with metadata filtering.

    The key difference from naive RAG:
    - Naive RAG: embed query → search ALL chunks → return top 5
    - Our approach: embed query → filter by ticker/year/section → search WITHIN that subset

    Why does this matter for finance?
    If you have 5 companies × 3 years = 15 documents in your vector store,
    a naive search for "revenue risk" might return chunks from the wrong
    company or wrong year. Metadata filtering prevents this entirely.
    """

    def __init__(self, collection_name: str = "finsight"):
        self.client = OpenAI()
        self.store = VectorStore(collection_name=collection_name)

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
        Embeds the query and retrieves the most relevant chunks.

        Args:
            query:              The search query (usually the user's question)
            ticker:             Filter to this company only
            year:               Filter to this fiscal year only
            n_results:          How many chunks to return
            section_filter:     Optional — narrow to a specific SEC section
            chunk_type_filter:  Optional — return only tables or only text

        Returns:
            List of RetrievedChunk dicts ordered by relevance score
        """
        # Step 1: Embed the query using the same model used for chunks
        # CRITICAL: must use identical model — mixing models breaks similarity
        query_embedding = self._embed_query(query)

        # Step 2: Build metadata filter
        # ChromaDB uses MongoDB-style query operators
        filters = self._build_filters(ticker, year, section_filter, chunk_type_filter)

        # Step 3: Query vector store
        results = self.store.query(
            query_embedding=query_embedding,
            n_results=n_results,
            filters=filters,
        )

        return results

    def run_multi_section(
        self,
        query: str,
        ticker: str,
        year: int,
        sections: list[str],
        n_results_per_section: int = 3,
    ) -> list[RetrievedChunk]:
        """
        Retrieves from multiple sections and merges results.
        Used when a query spans multiple parts of the filing.

        Example: "What is Apple's revenue and what risks could affect it?"
        → needs chunks from both Financial Statements AND Risk Factors
        """
        all_results = []
        seen_ids = set()

        for section in sections:
            results = self.run(
                query=query,
                ticker=ticker,
                year=year,
                n_results=n_results_per_section,
                section_filter=section,
            )
            # Deduplicate — same chunk can appear in multiple section searches
            for r in results:
                if r["id"] not in seen_ids:
                    all_results.append(r)
                    seen_ids.add(r["id"])

        # Re-sort by score after merging
        all_results.sort(key=lambda x: x["score"], reverse=True)
        return all_results

    def _embed_query(self, query: str) -> list[float]:
        response = self.client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=query,
        )
        return response.data[0].embedding

    def _build_filters(
        self,
        ticker: str,
        year: int,
        section_filter: str = None,
        chunk_type_filter: str = None,
    ) -> dict:
        """
        Builds ChromaDB where clause.
        ChromaDB requires $and for multiple conditions.
        """
        conditions = [
            {"ticker": {"$eq": ticker}},
            {"year": {"$eq": year}},
        ]
        if section_filter:
            conditions.append({"section": {"$eq": section_filter}})
        if chunk_type_filter:
            conditions.append({"chunk_type": {"$eq": chunk_type_filter}})

        return {"$and": conditions} if len(conditions) > 1 else conditions[0]