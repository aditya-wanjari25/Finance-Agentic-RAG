# retrieval/opensearch_store.py

import os
import json
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth
import boto3
from dotenv import load_dotenv
import time

load_dotenv()

# Index name — equivalent to ChromaDB collection
INDEX_NAME = "finsight-chunks"

# Vector dimensions must match our embedding model
# text-embedding-3-small outputs 1536 dimensions
VECTOR_DIMS = 1536


class OpenSearchStore:
    """
    OpenSearch vector store — production replacement for ChromaDB.

    Key advantages over ChromaDB:
    - Persistent outside the container (survives restarts)
    - Native hybrid search: BM25 + vector in one query
    - Managed by AWS: backups, scaling, monitoring included
    - Standard in finance/enterprise environments

    Index structure:
    - content_vector: 1536-dim vector for semantic search
    - content: raw text for BM25 keyword search
    - metadata fields: ticker, year, section, page etc for filtering
    """

    def __init__(self):
        self.endpoint = os.getenv("OPENSEARCH_ENDPOINT")
        self.username = os.getenv("OPENSEARCH_USERNAME", "finsight-admin")
        self.password = os.getenv("OPENSEARCH_PASSWORD", "FinSight2025!")

        if not self.endpoint:
            raise ValueError("OPENSEARCH_ENDPOINT not set in .env")

        # Clean endpoint — remove https:// if present
        host = self.endpoint.replace("https://", "").replace("http://", "")

        self.client = OpenSearch(
            hosts=[{"host": host, "port": 443}],
            http_auth=(self.username, self.password),
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection,
        )

        # Create index if it doesn't exist
        self._create_index_if_not_exists()
        print(f"🔍 OpenSearch store ready: {INDEX_NAME}")

    def _create_index_if_not_exists(self):
        """
        Creates the OpenSearch index with the right mappings.

        Mappings define the schema — critical to get right upfront
        because you can't change vector dimensions after creation.

        knn_vector: enables approximate nearest neighbor search
        keyword fields: enable exact metadata filtering
        text fields: enable BM25 full-text search
        """
        if self.client.indices.exists(index=INDEX_NAME):
            return

        index_body = {
            "settings": {
                "index": {
                    "knn": True,            # enables vector search
                    "knn.algo_param.ef_search": 100,  # accuracy vs speed tradeoff
                }
            },
            "mappings": {
                "properties": {
                    # Vector field — must match embedding dimensions exactly
                    "content_vector": {
                        "type": "knn_vector",
                        "dimension": VECTOR_DIMS,
                        "method": {
                            "name": "hnsw",          # Hierarchical NSW algorithm
                            "space_type": "cosinesimil",
                            "engine": "nmslib",
                        }
                    },
                    # Text field for BM25 keyword search
                    "content": {"type": "text"},
                    # Metadata fields — keyword type for exact filtering
                    "ticker":      {"type": "keyword"},
                    "year":        {"type": "integer"},
                    "quarter":     {"type": "keyword"},
                    "section":     {"type": "keyword"},
                    "page":        {"type": "integer"},
                    "chunk_type":  {"type": "keyword"},
                    "source_file": {"type": "keyword"},
                    "chunk_index": {"type": "integer"},
                }
            }
        }

        self.client.indices.create(index=INDEX_NAME, body=index_body)
        print(f"✅ Created OpenSearch index: {INDEX_NAME}")

    def store(self, chunks: list, embeddings: list[list[float]]) -> None:
        """
        Stores chunks using batched bulk indexing with delays between batches
        to avoid overwhelming the t3.small instance.
        """

        if len(chunks) != len(embeddings):
            raise ValueError("Chunks and embeddings count must match")

        BATCH_SIZE = 50  # smaller batches for t3.small
        total = len(chunks)
        stored = 0

        for i in range(0, total, BATCH_SIZE):
            batch_chunks = chunks[i:i+BATCH_SIZE]
            batch_embeddings = embeddings[i:i+BATCH_SIZE]

            bulk_body = []
            for chunk, embedding in zip(batch_chunks, batch_embeddings):
                fmt = chunk.to_chroma_format()
                meta = fmt["metadata"]

                bulk_body.append({
                    "index": {
                        "_index": INDEX_NAME,
                        "_id": fmt["id"],
                    }
                })
                bulk_body.append({
                    "content": fmt["document"],
                    "content_vector": embedding,
                    "ticker": meta.get("ticker", ""),
                    "year": meta.get("year", 0),
                    "quarter": meta.get("quarter", "annual"),
                    "section": meta.get("section", "Unknown"),
                    "page": meta.get("page", 0),
                    "chunk_type": meta.get("chunk_type", "text"),
                    "source_file": meta.get("source_file", ""),
                })

            response = self.client.bulk(body=bulk_body)
            stored += len(batch_chunks)

            if response["errors"]:
                print(f"  ⚠️  Some errors in batch {i//BATCH_SIZE + 1}")
            else:
                print(f"  Batch {i//BATCH_SIZE + 1} done ✓ ({stored}/{total})")

            # Give t3.small time to breathe between batches
            if i + BATCH_SIZE < total:
                time.sleep(2)

        print(f"✅ Stored {stored} chunks in OpenSearch")

    def query(
        self,
        query_embedding: list[float],
        n_results: int = 5,
        filters: dict = None,
    ) -> list[dict]:
        """
        Hybrid search: combines vector similarity + BM25 keyword search.

        This is the key upgrade over ChromaDB — we get both semantic
        and keyword matching in a single query. Great for financial terms
        like exact ticker symbols, section names, and numerical figures.

        Uses Reciprocal Rank Fusion to merge the two result sets.
        """
        # Build filter clause for metadata
        filter_clause = self._build_filter(filters)

        # Hybrid query: knn vector search + BM25 text search
        query_body = {
            "size": n_results,
            "query": {
                "bool": {
                    "should": [
                        # Vector search component
                        {
                            "knn": {
                                "content_vector": {
                                    "vector": query_embedding,
                                    "k": n_results * 2,
                                }
                            }
                        },
                    ],
                    "filter": filter_clause,
                }
            }
        }

        response = self.client.search(
            index=INDEX_NAME,
            body=query_body,
        )

        return self._format_results(response)

    def _build_filter(self, filters: dict = None) -> list:
        """Converts our filter dict to OpenSearch filter clause."""
        if not filters:
            return []

        clauses = []
        for key, value in filters.items():
            if key.startswith("$"):
                continue  # skip ChromaDB-style operators
            clauses.append({"term": {key: value}})

        return clauses

    def _format_results(self, response: dict) -> list[dict]:
        """Formats OpenSearch response into our standard chunk format."""
        results = []
        hits = response.get("hits", {}).get("hits", [])

        for hit in hits:
            source = hit["_source"]
            results.append({
                "id": hit["_id"],
                "content": source.get("content", ""),
                "score": hit["_score"],
                "metadata": {
                    "ticker": source.get("ticker"),
                    "year": source.get("year"),
                    "quarter": source.get("quarter"),
                    "section": source.get("section"),
                    "page": source.get("page"),
                    "chunk_type": source.get("chunk_type"),
                    "source_file": source.get("source_file"),
                }
            })

        return results

    def get_by_metadata(
        self,
        ticker: str = None,
        year: int = None,
        section: str = None,
        limit: int = 50,
    ) -> list[dict]:
        """
        Fetches chunks by metadata only — no vector search.
        Used for summarization queries.
        """
        must_clauses = []
        if ticker:
            must_clauses.append({"term": {"ticker": ticker}})
        if year:
            must_clauses.append({"term": {"year": year}})
        if section:
            must_clauses.append({"term": {"section": section}})

        query_body = {
            "size": limit,
            "query": {
                "bool": {
                    "must": must_clauses if must_clauses else [{"match_all": {}}]
                }
            },
            "sort": [{"page": "asc"}],  # preserve document order
        }

        response = self.client.search(index=INDEX_NAME, body=query_body)
        return self._format_results(response)

    def get_stats(self) -> dict:
        """Returns index statistics using count API instead of cat API."""
        response = self.client.count(index=INDEX_NAME)
        return {
            "total_chunks": response["count"],
            "index": INDEX_NAME,
        }