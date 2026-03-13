# retrieval/reranker.py

from sentence_transformers import CrossEncoder
from agents.state import RetrievedChunk

# This model is small (22MB), runs on CPU, and is very good at
# judging relevance of (query, passage) pairs.
# It was trained on MS MARCO — a large passage ranking dataset.
# No API key needed — runs fully locally.
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    """
    Cross-encoder reranker for retrieved chunks.

    Pattern: retrieve many candidates cheaply → rerank accurately → return top-k

    Why this improves scores:
    - Cosine similarity: compares query vector vs chunk vector independently
    - Cross-encoder: reads query + chunk together, understands relevance in context

    The cross-encoder is ~100x slower per comparison but much more accurate.
    We only run it on top-20 candidates, so latency stays acceptable (~200ms).
    """

    def __init__(self):
        print("🔄 Loading reranker model (first run downloads ~22MB)...")
        self.model = CrossEncoder(RERANKER_MODEL)
        print("✅ Reranker ready")

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int = 6,
        min_score: float = 0.0,
    ) -> list[RetrievedChunk]:
        if not chunks:
            return []

        pairs = [(query, chunk["content"]) for chunk in chunks]
        scores = self.model.predict(pairs)

        for chunk, score in zip(chunks, scores):
            chunk["reranker_score"] = float(score)

        # Filter below threshold first, then sort
        filtered = [c for c in chunks if c["reranker_score"] >= min_score]

        # Fall back to all chunks if filter removes everything
        if not filtered:
            filtered = chunks

        reranked = sorted(filtered, key=lambda x: x["reranker_score"], reverse=True)
        top = reranked[:top_k]

        print(f"  📊 Reranker: {len(chunks)} candidates → "
            f"{len(filtered)} above threshold → top {len(top)} "
            f"(scores: {top[0]['reranker_score']:.3f} to {top[-1]['reranker_score']:.3f})")

        return top