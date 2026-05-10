"""CLI entry point for the agno-docs-agent MCP server.

Start the server with::

    python -m agno_docs_agent --docs-path /path/to/agno-docs --model openai:gpt-4o-mini

All flags can also be set via environment variables:
``AGNO_DOCS_PATH``, ``AGNO_MODEL``, ``AGNO_EMBED_MODEL``.
"""

from __future__ import annotations

import argparse
import os


def main() -> None:
    """Parse CLI arguments and start the FastMCP server."""
    parser = argparse.ArgumentParser(
        prog="agno-docs-agent",
        description="Agentic MCP expert for the Agno framework documentation.",
    )
    parser.add_argument(
        "--docs-path",
        default=os.environ.get("AGNO_DOCS_PATH", "."),
        help="Path to the agno-docs directory containing llms-full.txt",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("AGNO_MODEL", "ollama:gemma4:26b"),
        help="Model identifier (e.g. openai:gpt-4o-mini, ollama:llama3)",
    )
    parser.add_argument(
        "--embed-model",
        default=os.environ.get("AGNO_EMBED_MODEL", "all-MiniLM-L6-v2"),
        help="SentenceTransformer model for local embeddings",
    )

    args = parser.parse_args()

    # Push into environment for the server to read
    os.environ["AGNO_DOCS_PATH"] = args.docs_path
    os.environ["AGNO_MODEL"] = args.model
    os.environ["AGNO_EMBED_MODEL"] = args.embed_model

    from agno_docs_agent.server import run_server

    run_server()


if __name__ == "__main__":
    main()
