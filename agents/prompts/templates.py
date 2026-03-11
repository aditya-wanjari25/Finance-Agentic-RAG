# agents/prompts/templates.py

QUERY_ANALYSIS_TEMPLATE = """Analyze the following financial query and extract structured information.

Query: {query}
Company: {ticker}
Year: {year}

Determine:
1. query_type: one of "retrieval", "comparison", "calculation", "summary"
2. section_filter: the most relevant SEC section to search in, or null if the whole document
   Options: "Risk Factors", "MD&A", "Financial Statements", "Business", "Properties",
            "Legal Proceedings", "Controls and Procedures", "Market for Registrant"
3. comparison_year: if this is a comparison query, what second year is being compared? else null

Respond in this exact JSON format with no other text:
{{
  "query_type": "retrieval",
  "section_filter": "Risk Factors",
  "comparison_year": null
}}"""


GENERATION_TEMPLATE = """Answer the following financial question using ONLY the retrieved context below.

## Question
{query}

## Company & Filing
Ticker: {ticker} | Year: {year} | Filing: {quarter}

## Retrieved Context
{context}

## Tool Results (if any)
{tool_results}

## Instructions
- Answer based strictly on the context provided
- Cite every factual claim with [Section, Page X]
- If context is insufficient, explicitly state what information is missing
- For numerical claims, always state the unit and time period
- Clearly label any interpretation as "Analysis:"
"""


COMPARISON_TEMPLATE = """Compare the following information across time periods.

## Question
{query}

## Period 1 Context ({year})
{context_current}

## Period 2 Context ({comparison_year})
{context_comparison}

## Instructions
- Create a structured comparison highlighting key differences
- Quantify changes where possible (e.g., "increased 12% from $X to $Y")
- Note any new risks or developments in the more recent period
- Cite every claim with [Section, Page X, Year]
"""