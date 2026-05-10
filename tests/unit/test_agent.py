"""Unit tests for the Agno agent factory.

Tests that create_agent() correctly wires model, instructions, knowledge, and tools
into an Agno Agent instance.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from agno_docs_agent.agent import create_agent
from agno_docs_agent.prompts.instructions import AGENT_SYSTEM_PROMPT


class FakeKnowledge:
    """Minimal knowledge stub with the interface the Agent expects."""

    def search(self, query: str, **kwargs) -> list:
        return []

    async def asearch(self, query: str, **kwargs) -> list:
        return []


class TestCreateAgent:
    """Verify the agent factory produces correctly configured Agno Agents."""

    @pytest.fixture
    def mock_mcp_tools(self) -> MagicMock:
        """Return a mocked MCPTools instance."""
        return MagicMock()

    @pytest.fixture
    def knowledge(self) -> FakeKnowledge:
        """Return a minimal knowledge implementation."""
        return FakeKnowledge()

    def test_agent_has_model_string(self, knowledge, mock_mcp_tools) -> None:
        """Agent MUST accept a model as a plain string identifier."""
        agent = create_agent(
            model="openai:gpt-4o-mini",
            instructions=AGENT_SYSTEM_PROMPT,
            knowledge=knowledge,
            mcp_tools=mock_mcp_tools,
        )
        assert agent.model is not None
        assert agent.model.id == "gpt-4o-mini"

    def test_agent_has_instructions(self, knowledge, mock_mcp_tools) -> None:
        """Agent MUST receive the system prompt as instructions."""
        agent = create_agent(
            model="openai:gpt-4o-mini",
            instructions=AGENT_SYSTEM_PROMPT,
            knowledge=knowledge,
            mcp_tools=mock_mcp_tools,
        )
        assert agent.instructions == AGENT_SYSTEM_PROMPT

    def test_agent_receives_knowledge(self, knowledge, mock_mcp_tools) -> None:
        """Agent MUST have the knowledge object injected."""
        agent = create_agent(
            model="openai:gpt-4o-mini",
            instructions=AGENT_SYSTEM_PROMPT,
            knowledge=knowledge,
            mcp_tools=mock_mcp_tools,
        )
        assert agent.knowledge is knowledge

    def test_agent_receives_tools(self, knowledge, mock_mcp_tools) -> None:
        """Agent MUST have MCP tools registered."""
        agent = create_agent(
            model="openai:gpt-4o",
            instructions=AGENT_SYSTEM_PROMPT,
            knowledge=knowledge,
            mcp_tools=mock_mcp_tools,
        )
        assert mock_mcp_tools in agent.tools

    def test_agent_has_name(self, knowledge, mock_mcp_tools) -> None:
        """Agent SHOULD have a descriptive name."""
        agent = create_agent(
            model="openai:gpt-4o",
            instructions=AGENT_SYSTEM_PROMPT,
            knowledge=knowledge,
            mcp_tools=mock_mcp_tools,
        )
        assert agent.name is not None
        assert "agno" in agent.name.lower()

    def test_agent_search_knowledge_enabled(self, knowledge, mock_mcp_tools) -> None:
        """Agent MUST have search_knowledge enabled for semantic RAG."""
        agent = create_agent(
            model="openai:gpt-4o-mini",
            instructions=AGENT_SYSTEM_PROMPT,
            knowledge=knowledge,
            mcp_tools=mock_mcp_tools,
        )
        assert agent.search_knowledge is True
