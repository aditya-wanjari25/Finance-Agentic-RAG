# ingestion/embedders/openai_embedder.py

import time
from openai import OpenAI
from dotenv import load_dotenv
from ingestion.chunkers.hierarchical_chunker import Chunk

load_dotenv()

# text-embedding-3-small: 1536 dimensions, cheap, fast
EMBEDDING_MODEL = "text-embedding-3-small"
BATCH_SIZE = 100  # OpenAI recommends batches of 100 for throughput


class OpenAIEmbedder:
    """
    Wraps OpenAI's embedding API with batching and retry logic.

    Why batching? Each API call has network overhead. Batching 100 chunks
    into one call is ~100x faster than individual calls and uses fewer
    rate limit tokens.

    Why retry logic? Embedding jobs on large documents can hit OpenAI's
    rate limits (especially on free tier). Simple exponential backoff
    handles this gracefully.
    """

    def __init__(self):
        self.client = OpenAI()
        self.model = EMBEDDING_MODEL
        self.total_tokens_used = 0

    def embed_chunks(self, chunks: list[Chunk]) -> list[list[float]]:
        """
        Takes a list of Chunks, returns a list of embedding vectors
        in the same order. Each vector is a list of 1536 floats.

        We embed chunk.content (which includes the context prefix)
        NOT chunk.raw_content — the prefix improves retrieval quality.
        """
        texts = [chunk.content for chunk in chunks]
        all_embeddings = []

        # Split into batches
        batches = [texts[i:i+BATCH_SIZE] for i in range(0, len(texts), BATCH_SIZE)]

        print(f"🔢 Embedding {len(chunks)} chunks in {len(batches)} batches...")

        for i, batch in enumerate(batches):
            embeddings = self._embed_with_retry(batch)
            all_embeddings.extend(embeddings)
            print(f"  Batch {i+1}/{len(batches)} done ✓")

            # Small sleep between batches to avoid rate limit bursts
            if i < len(batches) - 1:
                time.sleep(0.5)

        print(f"✅ Embedding complete. Approximate cost: "
              f"${self.total_tokens_used / 1_000_000 * 0.02:.4f}")

        return all_embeddings

    def _embed_with_retry(self, texts: list[str], max_retries: int = 3) -> list[list[float]]:
        """
        Calls OpenAI embedding API with exponential backoff on failure.
        Backoff: 2s → 4s → 8s between retries.
        """
        for attempt in range(max_retries):
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=texts,
                )
                # Track token usage for cost monitoring
                self.total_tokens_used += response.usage.total_tokens

                # Response comes back ordered by index — extract in order
                embeddings = [item.embedding for item in sorted(
                    response.data, key=lambda x: x.index
                )]
                return embeddings

            except Exception as e:
                wait = 2 ** (attempt + 1)
                print(f"  ⚠️  Embedding attempt {attempt+1} failed: {e}. "
                      f"Retrying in {wait}s...")
                time.sleep(wait)

        raise RuntimeError(f"Embedding failed after {max_retries} attempts")