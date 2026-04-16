# retrieval/hybrid_retriever.py
#
# Hybrid retrieval: BM25 (keyword) + vector (semantic) fused with Reciprocal Rank Fusion.
#
# Why hybrid?
# - Vector search excels at semantic similarity: "liquidity concerns" matches "cash flow risk"
# - BM25 excels at exact matches: "$394.3 billion", "Item 1A", specific ticker symbols
# - RRF combination consistently outperforms either alone without requiring score normalization
#
# Flow per query:
#   1. Fetch filtered corpus (all chunks for ticker+year+section) via get_by_metadata
#   2. Build BM25 index on corpus → rank by keyword relevance
#   3. Run vector search with same filters → rank by semantic relevance
#   4. Fuse both ranked lists with RRF → return top-k

from openai import OpenAI
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv
from retrieval.vector_store import get_vector_store
from agents.state import RetrievedChunk

load_dotenv()

EMBEDDING_MODEL = "text-embedding-3-small"

# Standard RRF constant from the original paper (Cormack et al., 2009).
# k=60 smooths rank differences — a doc at rank 1 vs rank 2 matters less
# than a doc at rank 1 vs rank 61. Tune down for sharper top-rank preference.
RRF_K = 60


def _tokenize(text: str) -> list[str]:
    """Whitespace tokenizer with lowercasing. Keeps numbers and symbols intact."""
    return text.lower().split()


def _reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    corpus_by_id: dict[str, dict],
) -> list[RetrievedChunk]:
    """
    Merges multiple ranked lists into one using RRF scores.

    For each document, score = Σ 1/(RRF_K + rank) across all lists.
    Documents that rank highly in multiple lists get the biggest boost.
    Documents absent from a list are simply not penalized for that list.
    """
    scores: dict[str, float] = {}
    for ranked_list in ranked_lists:
        for rank, doc_id in enumerate(ranked_list):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (RRF_K + rank + 1)

    sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)

    result = []
    for doc_id in sorted_ids:
        if doc_id in corpus_by_id:
            chunk = dict(corpus_by_id[doc_id])
            chunk["score"] = round(scores[doc_id], 6)
            result.append(chunk)

    return result


class HybridRetriever:
    """
    Combines BM25 and vector retrieval for a given metadata-filtered corpus.

    The BM25 index is built fresh per query from the filtered doc set.
    At the scale of a few hundred chunks per ticker/year/section this adds
    ~5-10ms — negligible compared to the embedding API call.
    """

    def __init__(self):
        self.client = OpenAI()
        self.store = get_vector_store()

    def retrieve(
        self,
        query: str,
        ticker: str,
        year: int,
        n_results: int = 5,
        section_filter: str = None,
        chunk_type_filter: str = None,
    ) -> list[RetrievedChunk]:
        """
        Runs hybrid retrieval and returns top-n chunks ranked by RRF score.

        Args:
            query:             The user's question
            ticker:            Filter to this company only
            year:              Filter to this fiscal year only
            n_results:         Number of chunks to return
            section_filter:    Optional — narrow to one SEC section
            chunk_type_filter: Optional — "text" or "table" only
        """
        # Step 1 — Fetch the filtered corpus for BM25
        # get_by_metadata returns ALL matching chunks (ordered by page),
        # giving BM25 the full picture of the relevant document subset.
        corpus = self.store.get_by_metadata(
            ticker=ticker,
            year=year,
            section=section_filter,
            limit=500,
        )

        if not corpus:
            return []

        if chunk_type_filter:
            corpus = [c for c in corpus if c["metadata"].get("chunk_type") == chunk_type_filter]

        if not corpus:
            return []

        corpus_by_id = {c["id"]: c for c in corpus}

        # Step 2 — BM25 ranking over the filtered corpus
        tokenized = [_tokenize(c["content"]) for c in corpus]
        bm25 = BM25Okapi(tokenized)
        bm25_scores = bm25.get_scores(_tokenize(query))
        bm25_ranked = [
            corpus[i]["id"]
            for i in sorted(range(len(corpus)), key=lambda x: bm25_scores[x], reverse=True)
        ]

        # Step 3 — Vector search with the same metadata filters
        query_embedding = self.client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=query,
        ).data[0].embedding

        filters = self._build_filters(ticker, year, section_filter, chunk_type_filter)
        vector_results = self.store.query(
            query_embedding=query_embedding,
            n_results=min(n_results * 4, len(corpus)),  # fetch extra candidates for RRF
            filters=filters,
        )
        vector_ranked = [r["id"] for r in vector_results]

        # Carry vector scores into corpus_by_id so they're visible in the result
        for r in vector_results:
            if r["id"] in corpus_by_id:
                corpus_by_id[r["id"]]["vector_score"] = r["score"]

        # Step 4 — Reciprocal Rank Fusion
        fused = _reciprocal_rank_fusion(
            ranked_lists=[vector_ranked, bm25_ranked],
            corpus_by_id=corpus_by_id,
        )

        print(f"  📊 Hybrid: {len(vector_results)} vector + {len(corpus)} BM25 → top {n_results} via RRF")

        return fused[:n_results]

    def _build_filters(
        self,
        ticker: str,
        year: int,
        section_filter: str = None,
        chunk_type_filter: str = None,
    ) -> dict:
        # Guard against None values — ChromaDB rejects them in where clauses
        conditions = []
        if ticker is not None:
            conditions.append({"ticker": {"$eq": ticker}})
        if year is not None:
            conditions.append({"year": {"$eq": year}})
        if section_filter is not None:
            conditions.append({"section": {"$eq": section_filter}})
        if chunk_type_filter is not None:
            conditions.append({"chunk_type": {"$eq": chunk_type_filter}})
        if not conditions:
            return {}
        return {"$and": conditions} if len(conditions) > 1 else conditions[0]
