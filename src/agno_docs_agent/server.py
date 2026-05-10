"""FastMCP server that hosts the consult_agno_expert tool.

On startup (lifespan), the server:
1. Loads the local embedding model.
2. Initializes or opens the sqlite-vec knowledge base.
3. Seeds the vector store from ``llms-full.txt`` if empty.
4. Builds an Agno Agent with knowledge + MCP tools.
5. Registers ``consult_agno_expert`` as an MCP tool.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Dict

from agno.agent import Agent
from agno.tools.mcp import MCPTools
from mcp.server.fastmcp import FastMCP

from agno_docs_agent.agent import create_agent
from agno_docs_agent.knowledge.embedder import EmbedderConfig, LocalEmbedder
from agno_docs_agent.knowledge.store import VectorStore, VectorStoreConfig
from agno_docs_agent.prompts.instructions import AGENT_SYSTEM_PROMPT
from agno_docs_agent.tools.expert import ConsultInput, consult_agno_expert

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_LLMS_PATH: str = str(
    Path(__file__).resolve().parent.parent.parent / "knowledge" / "llms-full.txt"
)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(server: FastMCP) -> AsyncIterator[Dict[str, Any]]:
    """Seed the knowledge base and wire up the agent on server start.

    Yields a state dict containing the configured agent, which the tool
    accesses via ``_agent`` injection.
    """
    # Resolve paths
    docs_path = Path(os.environ.get("AGNO_DOCS_PATH", ".")).resolve()
    llms_path = docs_path / "llms-full.txt"
    db_path = docs_path / "knowledge.db"
    model_str = os.environ.get("AGNO_MODEL", "ollama/gemma4:26b")
    embed_model = os.environ.get("AGNO_EMBED_MODEL", "all-MiniLM-L6-v2")

    # 1. Load embedder
    embedder = LocalEmbedder(EmbedderConfig(model_name=embed_model))

    # 2. Initialize vector store
    store = VectorStore(VectorStoreConfig(db_path=str(db_path), dimensions=384))
    store.initialize()

    # 3. Seed if empty
    _seed_knowledge_base(store, embedder, llms_path)

    # 4. Connect to agno-docs-mcp
    mcp_cmd = f"python -m mcp_agno_docs {docs_path}"
    mcp_tools = MCPTools(command=mcp_cmd)

    # 5. Build agent
    agent: Agent = create_agent(
        model=model_str,
        instructions=AGENT_SYSTEM_PROMPT,
        knowledge=store,
        mcp_tools=mcp_tools,
    )

    # 6. Register tool with agent injection
    @server.tool()
    async def consult_agno_expert_tool(query: str) -> Dict[str, Any]:
        validated = ConsultInput(query=query)
        return await consult_agno_expert(validated.query, _agent=agent)

    try:
        yield {"agent": agent}
    finally:
        # Cleanup would go here
        pass


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

mcp = FastMCP("agno-docs-agent", lifespan=lifespan)


def run_server() -> None:
    """Entry point: start the FastMCP server on stdio."""
    mcp.run(transport="stdio")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seed_knowledge_base(
    store: VectorStore, embedder: LocalEmbedder, llms_path: Path
) -> None:
    """Chunk and embed llms-full.txt into the vector store if empty.

    Args:
        store: An initialized VectorStore.
        embedder: A loaded LocalEmbedder.
        llms_path: Path to the llms-full.txt knowledge file.
    """
    if not llms_path.exists():
        return

    # Quick check: already seeded?
    existing = store.search([0.0] * 384, limit=1)
    if existing:
        return

    # Read and chunk
    text = llms_path.read_text(encoding="utf-8")
    chunks = _chunk_text(text, chunk_size=512, overlap=100)

    if not chunks:
        return

    # Embed in batches to manage memory
    batch_size = 32
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        embeddings = embedder.embed_batch(batch)
        store.insert(batch, embeddings)


def _chunk_text(text: str, chunk_size: int = 512, overlap: int = 100) -> list[str]:
    """Split *text* into roughly ``chunk_size``-word segments with overlap.

    Args:
        text: Full text content to chunk.
        chunk_size: Approximate number of words per chunk.
        overlap: Number of words to overlap between consecutive chunks.

    Returns:
        List of text chunks.
    """
    words = text.split()
    if len(words) <= chunk_size:
        return [text]

    chunks: list[str] = []
    step = max(1, chunk_size - overlap)
    for i in range(0, len(words), step):
        chunk_words = words[i : i + chunk_size]
        if not chunk_words:
            break
        chunks.append(" ".join(chunk_words))
    return chunks
