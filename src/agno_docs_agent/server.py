"""FastMCP server that hosts the consult_agno_expert tool.

On startup (lifespan), the server:
1. Loads the Agno SentenceTransformerEmbedder.
2. Initializes or opens the sqlite-vec knowledge base via SqliteVecDb.
3. Seeds the vector store from ``llms-full.txt`` if empty using Agno's
   Knowledge with MarkdownReader + MarkdownChunking.
4. Builds an Agno Agent with knowledge + MCP tools.
5. Registers ``consult_agno_expert`` as an MCP tool.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Dict

from agno.agent import Agent
from agno.knowledge.chunking.markdown import MarkdownChunking
from agno.knowledge.embedder.sentence_transformer import SentenceTransformerEmbedder
from agno.knowledge.knowledge import Knowledge
from agno.knowledge.reader.markdown_reader import MarkdownReader
from agno.tools.mcp import MCPTools
from mcp.server.fastmcp import FastMCP

from agno_docs_agent.agent import create_agent
from agno_docs_agent.knowledge.store import SqliteVecDb, SqliteVecDbConfig
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

    # 1. Load embedder (Agno native)
    embedder = SentenceTransformerEmbedder(id=embed_model, dimensions=384)

    # 2. Initialize vector store (Agno VectorDb ABC)
    store = SqliteVecDb(
        config=SqliteVecDbConfig(db_path=str(db_path), dimensions=384),
        embedder=embedder,
    )
    store.create()

    # 3. Build Agno Knowledge and seed if empty
    knowledge = Knowledge(
        vector_db=store,
        name="agno-docs",
    )
    await _load_knowledge(knowledge, llms_path)

    # 4. Connect to agno-docs-mcp
    mcp_cmd = f"python -m mcp_agno_docs {docs_path}"
    mcp_tools = MCPTools(command=mcp_cmd)

    # 5. Build agent
    agent: Agent = create_agent(
        model=model_str,
        instructions=AGENT_SYSTEM_PROMPT,
        knowledge=knowledge,
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


async def _load_knowledge(knowledge: Knowledge, llms_path: Path) -> None:
    """Load and index ``llms-full.txt`` into the knowledge base if empty.

    Uses Agno's native ``MarkdownReader`` with ``MarkdownChunking`` for
    chunking, and the configured embedder + VectorDb for storage.

    Args:
        knowledge: An Agno Knowledge instance with a VectorDb backend.
        llms_path: Path to the llms-full.txt knowledge file.
    """
    import asyncio

    if not llms_path.exists():
        return

    # Quick check: already seeded?
    store = knowledge.vector_db
    if store is not None and store.exists():
        import sqlite_vec as _sv  # type: ignore[import-untyped]
        zero_vec = _sv.serialize_float32([0.0] * 384)
        row = store._get_conn().execute(
            f"SELECT 1 FROM {store._config.table_name} "
            "WHERE embedding MATCH ? LIMIT 1",
            (zero_vec,),
        ).fetchone()
        if row is not None:
            return

    # Use Agno's MarkdownReader with MarkdownChunking for chunking
    await asyncio.to_thread(
        knowledge.insert,
        path=str(llms_path),
        reader=MarkdownReader(
            chunking_strategy=MarkdownChunking(chunk_size=512)
        ),
        upsert=True,
    )
