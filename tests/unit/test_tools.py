"""Unit tests for the consult_agno_expert MCP tool.

Tests Pydantic input validation, agent delegation, and error handling.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agno_docs_agent.tools.expert import ConsultInput, consult_agno_expert


class TestConsultInput:
    """Validate Pydantic input model constraints."""

    def test_valid_query_passes(self) -> None:
        """A valid query string MUST pass validation."""
        inp = ConsultInput(query="How does AgentOS work?")
        assert inp.query == "How does AgentOS work?"

    def test_query_too_short_fails(self) -> None:
        """A query shorter than 2 characters MUST fail validation."""
        with pytest.raises(ValueError):
            ConsultInput(query="x")

    def test_query_exceeds_max_length_fails(self) -> None:
        """A query longer than 500 characters MUST fail validation."""
        long_query = "x" * 501
        with pytest.raises(ValueError):
            ConsultInput(query=long_query)

    def test_query_within_max_length_passes(self) -> None:
        """A query exactly at 500 characters MUST pass."""
        max_query = "x" * 500
        inp = ConsultInput(query=max_query)
        assert len(inp.query) == 500


class TestConsultAgnoExpert:
    """Verify the tool delegates to the agent and returns structured output."""

    @pytest.fixture
    def mock_agent(self) -> MagicMock:
        """Return a mock Agno Agent with async run."""
        agent = MagicMock()
        agent.arun = AsyncMock()
        return agent

    @pytest.mark.asyncio
    async def test_tool_returns_expected_structure(self, mock_agent) -> None:
        """consult_agno_expert MUST return a dict with answer, sources, confidence."""
        mock_agent.arun.return_value = MagicMock(
            content="# Agno Architecture\n\nAgno uses AgentOS...\n\nSources: [docs](link)",
        )

        result = await consult_agno_expert("How does Agno work?", _agent=mock_agent)

        assert isinstance(result, dict)
        assert "answer" in result
        assert "sources" in result
        assert "confidence" in result
        assert isinstance(result["answer"], str)
        assert isinstance(result["sources"], list)
        assert isinstance(result["confidence"], float)
        assert 0.0 <= result["confidence"] <= 1.0

    @pytest.mark.asyncio
    async def test_tool_delegates_to_agent(self, mock_agent) -> None:
        """consult_agno_expert MUST forward the query to the agent's arun."""
        mock_agent.arun.return_value = MagicMock(content="The answer")

        await consult_agno_expert("What is AgentOS?", _agent=mock_agent)

        mock_agent.arun.assert_called_once()
        call_args = mock_agent.arun.call_args
        assert "What is AgentOS?" in str(call_args)

    @pytest.mark.asyncio
    async def test_tool_handles_agent_exception(self, mock_agent) -> None:
        """Agent failures MUST be wrapped as RuntimeError with context."""
        mock_agent.arun.side_effect = RuntimeError("Model unavailable")

        with pytest.raises(RuntimeError, match="Agent execution failed"):
            await consult_agno_expert("query", _agent=mock_agent)
