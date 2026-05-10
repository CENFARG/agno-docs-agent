"""MCP tool: consult_agno_expert — deep, cited answers about the Agno framework.

The tool receives a query, delegates to a configured Agno Agent, and returns a
structured response with an answer, source citations, and a confidence score.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class ConsultInput(BaseModel):
    """Input model for the consult_agno_expert MCP tool.

    Attributes:
        query: The Agno-related question. Must be between 2 and 500 characters.
    """

    query: str = Field(..., min_length=2, max_length=500, description="Your Agno question")


async def consult_agno_expert(
    query: str,
    _agent: Optional[Any] = None,
) -> Dict[str, Any]:
    """Answer a question about the Agno framework using hybrid search.

    Delegates to a pre-configured Agno Agent that combines semantic retrieval
    over the Agno reference knowledge base with live MCP doc tools.

    Args:
        query: The Agno-related question to answer.
        _agent: Internal agent reference injected by the server (not exposed
            to MCP clients).

    Returns:
        A dict with keys ``answer`` (str), ``sources`` (list[str]), and
        ``confidence`` (float between 0 and 1).

    Raises:
        RuntimeError: If the underlying agent encounters an unrecoverable error.
    """
    if _agent is None:
        raise RuntimeError("Agent not initialized — server must inject _agent")

    try:
        response = await _agent.arun(query)
    except Exception as exc:
        raise RuntimeError(f"Agent execution failed: {exc}") from exc

    content: str = getattr(response, "content", str(response))

    answer, sources, confidence = _parse_agent_response(content)
    return {"answer": answer, "sources": sources, "confidence": confidence}


def _parse_agent_response(
    content: str,
) -> tuple[str, list[str], float]:
    """Extract structured fields from the agent's raw response.

    Parses markdown-style agent output for citations and computes a simple
    confidence heuristic based on response length and source count.

    Args:
        content: Raw markdown string returned by the agent.

    Returns:
        A tuple of (answer_text, sources_list, confidence_score).
    """
    sources: list[str] = []
    # Naive extraction: look for URLs or doc paths in the response
    import re

    url_matches = re.findall(r"https?://[^\s)]+", content)
    doc_matches = re.findall(r"(?:docs/|/docs/|`/)[^\s`]+\.mdx?", content)
    sources.extend(url_matches)
    sources.extend(doc_matches)

    # Confidence heuristic: longer responses with citations = higher confidence
    base_confidence = min(1.0, len(content) / 2000.0)
    source_bonus = min(0.3, len(sources) * 0.1)
    confidence = round(min(1.0, base_confidence + source_bonus), 2)

    return content, sources, confidence
