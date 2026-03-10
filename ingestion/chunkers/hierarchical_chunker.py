# ingestion/chunkers/hierarchical_chunker.py

import re
import tiktoken
from dataclasses import dataclass
from typing import Optional
from ingestion.parsers.pdf_parser import ParsedBlock

# We use OpenAI's tokenizer since we're embedding with OpenAI models.
# cl100k_base is the encoding used by text-embedding-3-small and GPT-4o.
TOKENIZER = tiktoken.get_encoding("cl100k_base")

# Chunk size limits — tuned for finance documents.
# 512 tokens is a sweet spot: large enough for context, small enough for
# precise retrieval. Tables can be larger since we never split them.
MAX_TEXT_TOKENS = 512
MAX_TABLE_TOKENS = 1024  # tables can be bigger, LLM needs full context
CHUNK_OVERLAP_TOKENS = 50  # overlap between consecutive text chunks
MIN_CHUNK_TOKENS = 30  

@dataclass
class Chunk:
    """
    The output unit of our chunking stage.
    This is what gets embedded and stored in ChromaDB.

    content        — text fed to the embedding model (includes context prefix)
    raw_content    — original text without the prefix (what we show the user)
    metadata       — stored alongside the vector for filtering
    chunk_index    — global position in the document (for ordering results)
    token_count    — pre-computed so we don't re-tokenize later
    """
    content: str           # prefixed content (what gets embedded)
    raw_content: str       # clean content (what gets returned to user)
    chunk_index: int
    token_count: int
    metadata: dict

    def to_chroma_format(self) -> dict:
        """
        ChromaDB expects three separate lists:
        - documents: the text to embed
        - metadatas: the filter metadata
        - ids: unique string IDs

        This method packages a single chunk into that format.
        """
        return {
            "document": self.content,
            "metadata": self.metadata,
            "id": (f"{self.metadata['ticker']}_"
                   f"{self.metadata['year']}_"
                   f"chunk{self.chunk_index}")
        }

def count_tokens(text: str) -> int:
    """Count tokens using OpenAI's cl100k_base tokenizer."""
    return len(TOKENIZER.encode(text))


def split_by_paragraphs(text: str) -> list[str]:
    """
    Split text on double newlines (paragraph breaks).
    This is our first-preference split strategy because paragraphs
    are the natural unit of meaning in SEC filings.
    """
    paragraphs = re.split(r'\n\s*\n', text)
    return [p.strip() for p in paragraphs if p.strip()]


def split_by_sentences(text: str) -> list[str]:
    """
    Split text into sentences using punctuation patterns.
    Used when a single paragraph is still too large.
    We use regex rather than nltk to avoid extra dependencies.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def merge_small_chunks(chunks: list[str], max_tokens: int) -> list[str]:
    """
    After splitting, we often get tiny fragments (e.g. a one-line paragraph).
    This merges adjacent small chunks together up to max_tokens.
    Prevents storing useless 10-token chunks in our vector DB.
    """
    merged = []
    current = ""

    for chunk in chunks:
        candidate = (current + "\n\n" + chunk).strip() if current else chunk
        if count_tokens(candidate) <= max_tokens:
            current = candidate
        else:
            if current:
                merged.append(current)
            current = chunk

    if current:
        merged.append(current)

    return merged


TOC_SIGNALS = [
    r"item\s+1a?[\.\s]+.*\d+$",   # "Item 1A. Risk Factors ... 5"
    r"item\s+\d+[\.\s]+.*\d+$",   # "Item 7. MD&A ... 34"
]

def is_table_of_contents(markdown: str, page: int) -> bool:
    """
    Two-condition TOC detection:
    1. Table is in the first 5 pages (cover/TOC section of any 10-K)
    2. AND at least one row matches the Item X | Title | page_number pattern
    Both must be true — this prevents false positives on financial tables
    that happen to mention 'Item' in their content.
    """
    if page > 5:
        return False

    for line in markdown.split("\n"):
        if re.match(r"^\|\s*[-:]+\s*\|", line):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 2:
            continue
        first_cell = cells[0].lower()
        last_cell = cells[-1].strip()
        if "item" in first_cell and re.match(r"^\d{1,3}$", last_cell):
            return True  # even one match is enough given the page constraint

    return False

class HierarchicalChunker:
    """
    Converts a list of ParsedBlocks (from the parser) into a list of Chunks
    (ready for embedding and storage).

    Strategy:
    - Tables → always single chunks, never split
    - Headers → used to build context prefix, never stored alone
    - Text → split hierarchically: paragraph → sentence → character
    - Every chunk gets the section header prepended for better embeddings
    """

    def __init__(
        self,
        max_text_tokens: int = MAX_TEXT_TOKENS,
        max_table_tokens: int = MAX_TABLE_TOKENS,
    ):
        self.max_text_tokens = max_text_tokens
        self.max_table_tokens = max_table_tokens

    def chunk(self, blocks: list[ParsedBlock]) -> list[Chunk]:
        """
        Main entry point. Takes parser output, returns list of Chunks.
        """
        all_chunks = []
        chunk_index = 0
        current_section = "Unknown"

        for block in blocks:

            # Track the current section for context prefixing
            if block.block_type == "header":
                current_section = block.section
                continue  # headers are not stored as chunks

            elif block.block_type == "table":
                chunks = self._chunk_table(block, chunk_index)
                all_chunks.extend(chunks)
                chunk_index += len(chunks)

            elif block.block_type == "text":
                chunks = self._chunk_text(block, chunk_index)
                all_chunks.extend(chunks)
                chunk_index += len(chunks)

        print(f"✅ Chunking complete: {len(all_chunks)} chunks "
              f"({sum(1 for c in all_chunks if c.metadata['chunk_type'] == 'table')} tables, "
              f"{sum(1 for c in all_chunks if c.metadata['chunk_type'] == 'text')} text)")

        return all_chunks

    def _build_context_prefix(self, block: ParsedBlock) -> str:
        """
        Builds the context string prepended to every chunk before embedding.
        Example: "Company: AAPL | Year: 2025 | Section: MD&A\n\n"

        Why? Because when someone asks "what were Apple's 2025 revenue risks?",
        the embedding of that query will match chunks that contain this context.
        Without it, a chunk about revenue risks has no idea it belongs to AAPL 2025.
        """
        return (
            f"Company: {block.ticker} | "
            f"Year: {block.year} | "
            f"Section: {block.section}\n\n"
        )

    def _make_metadata(self, block: ParsedBlock, chunk_type: str) -> dict:
        """Builds the metadata dict stored in ChromaDB alongside the vector."""
        return {
            "ticker": block.ticker,
            "year": block.year,
            "quarter": block.quarter,
            "section": block.section,
            "page": block.page,
            "chunk_type": chunk_type,
            "source_file": block.source_file,
        }

    def _chunk_table(self, block: ParsedBlock, start_index: int) -> list[Chunk]:
        """
        Tables are always stored as a single chunk.
        If a table exceeds MAX_TABLE_TOKENS, we log a warning but still
        keep it whole — splitting a table destroys its meaning entirely.
        """
        # Skip table of contents — not useful for retrieval
        if is_table_of_contents(block.content, block.page):
            # print(f"  ⚠️  Skipping table of contents on page {block.page}")
            return []

        prefix = self._build_context_prefix(block)
        content = prefix + block.content
        token_count = count_tokens(content)

        if token_count > self.max_table_tokens:
            print(f"  ⚠️  Large table on page {block.page} "
                  f"({token_count} tokens) — keeping whole")

        return [Chunk(
            content=content,
            raw_content=block.content,
            chunk_index=start_index,
            token_count=token_count,
            metadata=self._make_metadata(block, "table"),
        )]

    def _chunk_text(self, block: ParsedBlock, start_index: int) -> list[Chunk]:
        """
        Splits a text block hierarchically:
        1. If it fits in max_text_tokens → single chunk
        2. Try splitting by paragraphs and merging small ones
        3. If any paragraph is still too big → split by sentences
        4. Last resort: hard character split with overlap
        """
        prefix = self._build_context_prefix(block)
        token_count = count_tokens(block.content)

        # Case 1: fits as-is
        if token_count <= self.max_text_tokens:
            # Skip noise chunks — too small to be meaningful
            if token_count < MIN_CHUNK_TOKENS:
                return []
            content = prefix + block.content
            return [Chunk(
                content=content,
                raw_content=block.content,
                chunk_index=start_index,
                token_count=count_tokens(content),
                metadata=self._make_metadata(block, "text"),
            )]

        # Case 2: split by paragraphs
        paragraphs = split_by_paragraphs(block.content)
        segments = merge_small_chunks(paragraphs, self.max_text_tokens)

        # Case 3: any segment still too large → split by sentences
        final_segments = []
        for segment in segments:
            if count_tokens(segment) <= self.max_text_tokens:
                final_segments.append(segment)
            else:
                sentences = split_by_sentences(segment)
                final_segments.extend(
                    merge_small_chunks(sentences, self.max_text_tokens)
                )

        # Case 4: last resort hard split (rarely triggered)
        chunks = []
        for i, segment in enumerate(final_segments):
            if count_tokens(segment) > self.max_text_tokens:
                segment = self._hard_split(segment)[0]  # take first piece

            content = prefix + segment
            chunks.append(Chunk(
                content=content,
                raw_content=segment,
                chunk_index=start_index + i,
                token_count=count_tokens(content),
                metadata=self._make_metadata(block, "text"),
            ))

        return chunks

    def _hard_split(self, text: str) -> list[str]:
        """
        Last resort: encode to tokens, split at max boundary, decode back.
        Guaranteed to produce chunks within token limits.
        Overlap helps preserve context across boundaries.
        """
        tokens = TOKENIZER.encode(text)
        segments = []
        start = 0

        while start < len(tokens):
            end = min(start + self.max_text_tokens, len(tokens))
            segment_tokens = tokens[start:end]
            segments.append(TOKENIZER.decode(segment_tokens))
            start = end - CHUNK_OVERLAP_TOKENS  # overlap

        return segments