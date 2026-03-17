# ingestion/pipeline.py

import os
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
    upload_to_s3: bool = False,
) -> dict:
    """
    Full ingestion pipeline: PDF → parsed blocks → chunks → embeddings → ChromaDB.

    Args:
        pdf_path:       Local path to PDF file
        ticker:         Stock ticker e.g. "AAPL"
        year:           Fiscal year e.g. 2025
        quarter:        "annual", "Q1", "Q2", "Q3"
        collection_name: ChromaDB collection to store in
        upload_to_s3:   If True, also uploads the PDF to S3 for cloud storage
    """
    print(f"\n{'='*50}")
    print(f"📄 Ingesting {ticker} {year} {quarter}")
    print(f"{'='*50}\n")

    # Optionally upload to S3 first
    if upload_to_s3:
        try:
            from ingestion.storage.s3_client import S3DocumentStore
            s3 = S3DocumentStore()
            s3_key = s3.upload(pdf_path, ticker=ticker, year=year)
            print(f"☁️  Backed up to S3: {s3_key}\n")
        except Exception as e:
            print(f"⚠️  S3 upload failed (continuing with local): {e}\n")

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


def ingest_from_s3(
    ticker: str,
    year: int,
    filename: str,
    quarter: str = "annual",
    collection_name: str = "finsight",
) -> dict:
    """
    Ingestion pipeline that pulls the PDF from S3 instead of local disk.
    This is what the ECS container will use in production.

    The PDF is downloaded to a temp file, processed, then the temp file
    is deleted — keeping the container stateless.
    """
    from ingestion.storage.s3_client import S3DocumentStore

    print(f"\n{'='*50}")
    print(f"📄 Ingesting from S3: {ticker} {year} {quarter}")
    print(f"{'='*50}\n")

    s3 = S3DocumentStore()
    temp_path = None

    try:
        # Download to temp file
        temp_path = s3.download_to_temp(
            ticker=ticker,
            year=year,
            filename=filename,
        )

        # Run standard ingestion on temp file
        summary = ingest_document(
            pdf_path=temp_path,
            ticker=ticker,
            year=year,
            quarter=quarter,
            collection_name=collection_name,
            upload_to_s3=False,  # already in S3
        )
        return summary

    finally:
        # Always clean up temp file
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)
            print(f"🧹 Cleaned up temp file: {temp_path}")


if __name__ == "__main__":
    ingest_document(
        pdf_path="data/raw/AAPL_10K_2025.pdf",
        ticker="AAPL",
        year=2025,
        upload_to_s3=True,  # now uploads to S3 as well
    )