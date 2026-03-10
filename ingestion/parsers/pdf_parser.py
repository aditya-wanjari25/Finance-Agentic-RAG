import re
import fitz                  # PyMuPDF — fast text extraction with position data
import pdfplumber            # Best-in-class table extraction
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

@dataclass
class ParsedBlock:
    """
    The atomic unit of our parsed document.
    Every piece of content — text, table, header — becomes one of these.
    Downstream chunking and embedding consumes a list of these.
    """
    content: str                        # The actual text or markdown table
    block_type: str                     # "text" | "table" | "header"
    page: int                           # 1-indexed page number
    section: str = "Unknown"            # e.g. "Risk Factors", "MD&A"
    ticker: str = ""                    # e.g. "AAPL"
    year: int = 0                       # e.g. 2023
    quarter: str = "annual"            # "annual" | "Q1" | "Q2" | "Q3"
    source_file: str = ""               # original filename

    def to_metadata(self) -> dict:
        """
        Returns a flat dict — this is what gets stored in ChromaDB alongside
        the embedding. Enables metadata filtering during retrieval.
        """
        return {
            "block_type": self.block_type,
            "page": self.page,
            "section": self.section,
            "ticker": self.ticker,
            "year": self.year,
            "quarter": self.quarter,
            "source_file": self.source_file,
        }

# 10-K filings follow a standard SEC structure with these exact item labels.
# We use these to track "where are we in the document" as we parse page by page.
# This is finance-domain knowledge baked into our parser.
SEC_SECTION_PATTERNS = [
    (r"item\s+1a[\.\s]", "Risk Factors"),
    (r"item\s+1b[\.\s]", "Unresolved Staff Comments"),
    (r"item\s+1[\.\s]",  "Business"),
    (r"item\s+2[\.\s]",  "Properties"),
    (r"item\s+3[\.\s]",  "Legal Proceedings"),
    (r"item\s+5[\.\s]",  "Market for Registrant"),
    (r"item\s+7a[\.\s]", "Quantitative Disclosures"),
    (r"item\s+7[\.\s]",  "MD&A"),
    (r"item\s+8[\.\s]",  "Financial Statements"),
    (r"item\s+9a[\.\s]", "Controls and Procedures"),
    (r"item\s+9[\.\s]",  "Changes in Disagreements"),
    (r"item\s+10[\.\s]", "Directors and Officers"),
    (r"item\s+11[\.\s]", "Executive Compensation"),
    (r"item\s+12[\.\s]", "Security Ownership"),
    (r"item\s+13[\.\s]", "Certain Relationships"),
    (r"item\s+14[\.\s]", "Principal Accountant Fees"),
]


def detect_section(text: str, current_section: str) -> str:
    """
    Given a line of text, check if it matches a known SEC section header.
    If yes, return the new section name. If no, return the current section unchanged.

    We lowercase and strip the text before matching because real PDFs have
    inconsistent casing and whitespace (e.g. "ITEM  1A." vs "Item 1a.").
    """
    cleaned = text.lower().strip()
    for pattern, section_name in SEC_SECTION_PATTERNS:
        if re.match(pattern, cleaned):
            return section_name
    return current_section


def extract_tables_from_page(page) -> list[tuple[str, list]]:
    """
    Uses pdfplumber to find all tables on a given page.
    Returns a list of (markdown_string, bounding_boxes) tuples.

    Why convert to markdown?
    Because markdown tables are human-readable, LLM-friendly,
    and preserve the row/column relationships that raw text destroys.

    Why track bounding boxes?
    So we can SKIP these regions during text extraction — otherwise
    pdfplumber and PyMuPDF would both extract the same table content,
    giving us duplicates.
    """
    tables_markdown = []

    for table in page.find_tables():
        rows = table.extract()
        if not rows:
            continue

        # Filter out completely empty rows — common in PDFs with decorative lines
        rows = [r for r in rows if any(cell and str(cell).strip() for cell in r)]
        if not rows:
            continue

        # Build a clean markdown table
        # Row 0 is treated as the header row
        md_lines = []
        header = [str(cell or "").strip() for cell in rows[0]]
        md_lines.append("| " + " | ".join(header) + " |")
        md_lines.append("| " + " | ".join(["---"] * len(header)) + " |")

        for row in rows[1:]:
            cells = [str(cell or "").strip() for cell in row]
            # Pad row if it has fewer cells than header (malformed tables)
            while len(cells) < len(header):
                cells.append("")
            md_lines.append("| " + " | ".join(cells) + " |")

        markdown = "\n".join(md_lines)
        tables_markdown.append((markdown, table.bbox))  # bbox = (x0, y0, x1, y1)

    return tables_markdown

class FinancialPDFParser:
    """
    Main parser class. Takes a PDF + document metadata, returns a list of
    ParsedBlocks ready for the chunking stage.

    Design decision: we do one pass with pdfplumber (table-aware, slower)
    and extract text blocks using PyMuPDF (fast, position-aware).
    We cross-reference bounding boxes so tables and text don't overlap.
    """

    def __init__(self, ticker: str, year: int, quarter: str = "annual"):
        self.ticker = ticker
        self.year = year
        self.quarter = quarter

    def parse(self, pdf_path: str) -> list[ParsedBlock]:
        """
        Main entry point. Feed it a path to a 10-K PDF.
        Returns an ordered list of ParsedBlocks preserving document flow.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        blocks = []
        current_section = "Unknown"

        # Open the same file with both libraries simultaneously
        # pdfplumber for tables, fitz for text
        with pdfplumber.open(pdf_path) as plumber_doc:
            fitz_doc = fitz.open(str(pdf_path))

            for page_num in range(len(fitz_doc)):
                plumber_page = plumber_doc.pages[page_num]
                fitz_page = fitz_doc[page_num]

                # --- Step 1: Extract tables first, get their bounding boxes ---
                tables_on_page = extract_tables_from_page(plumber_page)
                table_bboxes = [bbox for _, bbox in tables_on_page]

                # --- Step 2: Extract text blocks via PyMuPDF ---
                # "blocks" in PyMuPDF = a rectangle of text grouped together
                # Each block: (x0, y0, x1, y1, text, block_no, block_type)
                text_blocks = fitz_page.get_text("blocks")

                for block in text_blocks:
                    x0, y0, x1, y1, text, *_ = block
                    text = text.strip()
                    if not text or len(text) < 10:  # skip noise/whitespace
                        continue

                    # Skip this text block if it overlaps with a table region
                    # This prevents double-extraction of table content
                    if self._overlaps_table(x0, y0, x1, y1, table_bboxes):
                        continue

                    # Check if this block is a section header
                    first_line = text.split("\n")[0]
                    new_section = detect_section(first_line, current_section)
                    if new_section != current_section:
                        current_section = new_section
                        # Emit the header as its own block so we can use it
                        # for context injection later
                        blocks.append(ParsedBlock(
                            content=first_line,
                            block_type="header",
                            page=page_num + 1,
                            section=current_section,
                            ticker=self.ticker,
                            year=self.year,
                            quarter=self.quarter,
                            source_file=pdf_path.name,
                        ))
                        # Remove the header line, keep the rest as text
                        remaining = "\n".join(text.split("\n")[1:]).strip()
                        if remaining:
                            text = remaining
                        else:
                            continue

                    blocks.append(ParsedBlock(
                        content=text,
                        block_type="text",
                        page=page_num + 1,
                        section=current_section,
                        ticker=self.ticker,
                        year=self.year,
                        quarter=self.quarter,
                        source_file=pdf_path.name,
                    ))

                # --- Step 3: Add table blocks ---
                for markdown_table, _ in tables_on_page:
                    blocks.append(ParsedBlock(
                        content=markdown_table,
                        block_type="table",
                        page=page_num + 1,
                        section=current_section,
                        ticker=self.ticker,
                        year=self.year,
                        quarter=self.quarter,
                        source_file=pdf_path.name,
                    ))

            fitz_doc.close()

        print(f"✅ Parsed {pdf_path.name}: {len(blocks)} blocks "
              f"({sum(1 for b in blocks if b.block_type == 'table')} tables, "
              f"{sum(1 for b in blocks if b.block_type == 'text')} text, "
              f"{sum(1 for b in blocks if b.block_type == 'header')} headers)")

        return blocks

    def _overlaps_table(self, x0, y0, x1, y1, table_bboxes: list, threshold=0.5) -> bool:
        """
        Returns True if a text block significantly overlaps with any table region.
        We use a threshold (default 50% overlap) rather than any overlap,
        because sometimes text captions sit near table borders.
        """
        block_area = max((x1 - x0) * (y1 - y0), 1)

        for (tx0, ty0, tx1, ty1) in table_bboxes:
            # Calculate intersection area
            ix0, iy0 = max(x0, tx0), max(y0, ty0)
            ix1, iy1 = min(x1, tx1), min(y1, ty1)
            if ix1 <= ix0 or iy1 <= iy0:
                continue  # no intersection
            intersection = (ix1 - ix0) * (iy1 - iy0)
            if intersection / block_area > threshold:
                return True
        return False