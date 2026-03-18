# retrieval/vector_store.py

import chromadb
from chromadb.config import Settings
from ingestion.chunkers.hierarchical_chunker import Chunk
import os

# Persist to disk so embeddings survive between runs.
# Without this, ChromaDB resets every time — you'd re-embed on every run.
CHROMA_PATH = ".chroma"


class VectorStore:
    """
    ChromaDB wrapper for storing and querying chunk embeddings.
    """

    def __init__(self, collection_name: str = "finsight"):
        self.client = chromadb.PersistentClient(
            path=CHROMA_PATH,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        print(
            f"📦 Vector store ready. "
            f"Collection '{collection_name}' has "
            f"{self.collection.count()} existing chunks."
        )

    def store(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
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

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

        print(
            f"✅ Stored {len(chunks)} chunks. "
            f"Collection total: {self.collection.count()}"
        )

    def query(
        self,
        query_embedding: list[float],
        n_results: int = 5,
        filters: dict = None,
    ) -> list[dict]:
        query_params = {
            "query_embeddings": [query_embedding],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"],
        }

        if filters:
            query_params["where"] = filters

        results = self.collection.query(**query_params)

        chunks_out = []
        for i in range(len(results["ids"][0])):
            chunks_out.append(
                {
                    "id": results["ids"][0][i],
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "score": 1 - results["distances"][0][i],
                }
            )

        return chunks_out

    def get_collection_stats(self) -> dict:
        count = self.collection.count()
        return {"total_chunks": count, "collection": self.collection.name}

    def get_by_metadata(
        self,
        ticker: str = None,
        year: int = None,
        section: str = None,
        filters: dict = None,
        limit: int = 50,
    ) -> list[dict]:
        """
        Fetches chunks by metadata.
        """
        get_params = {
            "limit": limit,
            "include": ["documents", "metadatas"],
        }

        if ticker or year or section:
            conditions = []
            if ticker:
                conditions.append({"ticker": {"$eq": ticker}})
            if year:
                conditions.append({"year": {"$eq": year}})
            if section:
                conditions.append({"section": {"$eq": section}})

            get_params["where"] = (
                {"$and": conditions} if len(conditions) > 1 else conditions[0]
            )
        elif filters:
            get_params["where"] = filters

        results = self.collection.get(**get_params)

        chunks_out = []
        for i in range(len(results["ids"])):
            chunks_out.append(
                {
                    "id": results["ids"][i],
                    "content": results["documents"][i],
                    "metadata": results["metadatas"][i],
                    "score": 1.0,
                }
            )

        chunks_out.sort(key=lambda x: x["metadata"].get("page", 0))
        return chunks_out


def get_vector_store():
    """
    Factory function — returns the right vector store based on environment.
    """
    use_opensearch = os.getenv("USE_OPENSEARCH", "false").lower() == "true"

    if use_opensearch:
        from retrieval.opensearch_store import OpenSearchStore

        return OpenSearchStore()
    else:
        return VectorStore()