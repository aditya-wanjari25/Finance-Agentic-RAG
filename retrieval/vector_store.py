# retrieval/vector_store.py

import chromadb
from chromadb.config import Settings
from ingestion.chunkers.hierarchical_chunker import Chunk

# Persist to disk so embeddings survive between runs.
# Without this, ChromaDB resets every time — you'd re-embed on every run.
CHROMA_PATH = ".chroma"


class VectorStore:
    """
    ChromaDB wrapper for storing and querying chunk embeddings.

    ChromaDB organizes data into "collections" — think of each collection
    as a table in a database. We use one collection per project.

    Key operations:
    - store(): save chunks + their embeddings
    - query(): find most similar chunks to a query embedding
    - filter by metadata (ticker, year, section, chunk_type)
    """

    def __init__(self, collection_name: str = "finsight"):
        # PersistentClient saves to disk at CHROMA_PATH
        self.client = chromadb.PersistentClient(
            path=CHROMA_PATH,
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            # cosine distance is standard for text embeddings —
            # it measures angle between vectors, not magnitude
            metadata={"hnsw:space": "cosine"}
        )
        print(f"📦 Vector store ready. "
              f"Collection '{collection_name}' has "
              f"{self.collection.count()} existing chunks.")

    def store(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """
        Stores chunks and their embeddings in ChromaDB.

        ChromaDB expects parallel lists:
        - ids: unique string per chunk
        - embeddings: the vectors
        - documents: the text (stored for retrieval)
        - metadatas: filter fields

        We upsert (insert or update) so re-running the pipeline
        on the same document doesn't create duplicates.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Chunks ({len(chunks)}) and embeddings ({len(embeddings)}) must match"
            )

        ids, documents, metadatas = [], [], []

        for chunk, embedding in zip(chunks, embeddings):
            fmt = chunk.to_chroma_format()
            ids.append(fmt["id"])
            documents.append(fmt["document"])
            metadatas.append(fmt["metadata"])

        # Batch upsert — ChromaDB handles large lists efficiently
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        print(f"✅ Stored {len(chunks)} chunks. "
              f"Collection total: {self.collection.count()}")

    def query(
        self,
        query_embedding: list[float],
        n_results: int = 5,
        filters: dict = None,
    ) -> list[dict]:
        """
        Finds the n most similar chunks to a query embedding.

        filters: optional ChromaDB 'where' clause for metadata filtering.
        Examples:
            {"ticker": "AAPL"}
            {"$and": [{"ticker": "AAPL"}, {"year": 2025}]}
            {"section": "MD&A"}

        Returns a clean list of dicts with content + metadata + score.
        """
        query_params = {
            "query_embeddings": [query_embedding],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"],
        }
        if filters:
            query_params["where"] = filters

        results = self.collection.query(**query_params)

        # Reformat ChromaDB's nested response into a clean flat list
        chunks_out = []
        for i in range(len(results["ids"][0])):
            chunks_out.append({
                "id": results["ids"][0][i],
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                # ChromaDB returns cosine distance (0=identical, 2=opposite)
                # Convert to similarity score (1=identical, -1=opposite)
                "score": 1 - results["distances"][0][i],
            })

        return chunks_out

    def get_collection_stats(self) -> dict:
        """Useful for debugging — shows what's in the collection."""
        count = self.collection.count()
        return {"total_chunks": count, "collection": self.collection.name}
    
    def get_by_metadata(
    self,
    filters: dict = None,
    limit: int = 50,
    ) -> list[dict]:
        """
        Fetches chunks by metadata filter only — no vector similarity.
        Used for summarization where we want ALL chunks from a section,
        not just the semantically closest ones.
        """
        get_params = {
        "limit": limit,
        "include": ["documents", "metadatas"],
    }
        if filters:
            get_params["where"] = filters

        results = self.collection.get(**get_params)
        # results = self.collection.get(
        #     where=filters,
        #     limit=limit,
        #     include=["documents", "metadatas"],
        # )

        chunks_out = []
        for i in range(len(results["ids"])):
            chunks_out.append({
                "id": results["ids"][i],
                "content": results["documents"][i],
                "metadata": results["metadatas"][i],
                "score": 1.0,  # no similarity score for metadata-only fetch
            })

        # Sort by page number to preserve document flow
        chunks_out.sort(key=lambda x: x["metadata"].get("page", 0))
        return chunks_out