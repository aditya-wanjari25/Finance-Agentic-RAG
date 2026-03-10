# ingestion/pipeline.py

from ingestion.parsers.pdf_parser import FinancialPDFParser
from ingestion.chunkers.hierarchical_chunker import HierarchicalChunker
from ingestion.embedders.openai_embedder import OpenAIEmbedder
from retrieval.vector_store import VectorStore


def ingest_document(
    pdf_path: str,
    ticker: str,
    year: int,
    quarter: str = "annual",
    collection_name: str = "finsight",
) -> dict:
    """
    Full ingestion pipeline: PDF → parsed blocks → chunks → embeddings → ChromaDB.

    This is the single entry point for adding a new document to the system.
    Run this once per document — results persist to disk in .chroma/

    Args:
        pdf_path:    Path to the PDF file
        ticker:      Stock ticker e.g. "AAPL"
        year:        Fiscal year e.g. 2025
        quarter:     "annual", "Q1", "Q2", "Q3"
        collection_name: ChromaDB collection to store in

    Returns:
        Summary dict with counts and stats
    """
    print(f"\n{'='*50}")
    print(f"📄 Ingesting {ticker} {year} {quarter}")
    print(f"{'='*50}\n")

    # Step 1 — Parse
    parser = FinancialPDFParser(ticker=ticker, year=year, quarter=quarter)
    blocks = parser.parse(pdf_path)

    # Step 2 — Chunk
    chunker = HierarchicalChunker()
    chunks = chunker.chunk(blocks)

    # Step 3 — Embed
    embedder = OpenAIEmbedder()
    embeddings = embedder.embed_chunks(chunks)

    # Step 4 — Store
    store = VectorStore(collection_name=collection_name)
    store.store(chunks, embeddings)

    summary = {
        "ticker": ticker,
        "year": year,
        "quarter": quarter,
        "blocks_parsed": len(blocks),
        "chunks_stored": len(chunks),
        "tokens_used": embedder.total_tokens_used,
    }

    print(f"\n✅ Ingestion complete: {summary}")
    return summary


if __name__ == "__main__":
    # Run directly: python -m ingestion.pipeline
    ingest_document(
        pdf_path="data/raw/AAPL_10K_2025.pdf",
        ticker="AAPL",
        year=2025,
    )