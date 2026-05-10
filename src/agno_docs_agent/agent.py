"""Agent factory that wires an Agno Agent with knowledge, MCP tools, and instructions.

Provides a single ``create_agent()`` entry point that the server uses during
startup to configure the consult_agno_expert agent.
"""

from __future__ import annotations

from agno.agent import Agent
from agno.knowledge.protocol import KnowledgeProtocol
from agno.tools.mcp import MCPTools


def create_agent(
    model: str,
    instructions: str,
    knowledge: KnowledgeProtocol,
    mcp_tools: MCPTools,
) -> Agent:
    """Build a configured Agno Agent for answering Agno documentation questions.

    Args:
        model: A model identifier string (e.g. ``"openai:gpt-4o-mini"``)
            or an Agno ``Model`` object. This controls which LLM provider
            the agent uses.
        instructions: The system prompt that defines the agent's identity,
            workflow, and constraints.
        knowledge: A :class:`KnowledgeProtocol` implementation providing
            semantic retrieval capabilities.
        mcp_tools: An :class:`MCPTools` instance connected to an external
            MCP server (typically ``agno-docs-mcp``).

    Returns:
        A fully configured :class:`Agent` ready for ``agent.run()`` calls.
    """
    return Agent(
        model=model,
        name="AgnoDocsExpert",
        instructions=instructions,
        knowledge=knowledge,
        tools=[mcp_tools],
        search_knowledge=True,
        markdown=True,
    )
