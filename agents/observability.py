# agents/observability.py

import os
from dotenv import load_dotenv

load_dotenv()


def get_langsmith_config(
    run_name: str = "finsight-query",
    tags: list[str] = None,
    metadata: dict = None,
) -> dict:
    """
    Returns LangSmith run configuration for passing to agent invocations.

    Tags and metadata appear in the LangSmith UI and enable filtering.
    For example, filter by ticker='AAPL' or query_type='calculation'.

    Args:
        run_name:  Display name in LangSmith UI
        tags:      List of string tags e.g. ['production', 'AAPL']
        metadata:  Dict of key-value pairs shown in LangSmith UI
    """
    if not os.getenv("LANGCHAIN_TRACING_V2"):
        return {}

    config = {"run_name": run_name}

    if tags:
        config["tags"] = tags
    if metadata:
        config["metadata"] = metadata

    return config


def is_tracing_enabled() -> bool:
    """Returns True if LangSmith tracing is configured and enabled."""
    return (
        os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
        and bool(os.getenv("LANGSMITH_API_KEY"))
    )