"""Vector store backed by sqlite-vec for local semantic search.

Provides a lightweight, file-based vector database using the sqlite-vec extension
with vec0 virtual tables. Implements :class:`KnowledgeProtocol` so it can be used
directly as an Agno Agent's ``knowledge`` source.
"""

from __future__ import annotations

import asyncio
import sqlite3
from dataclasses import dataclass
from typing import Any, Callable, Dict, List

import sqlite_vec  # type: ignore[import-untyped]
from agno.knowledge.document import Document


@dataclass(frozen=True)
class VectorStoreConfig:
    """Configuration for the sqlite-vec vector store.

    Attributes:
        db_path: Filesystem path to the SQLite database file.
        table_name: Name of the vec0 virtual table (default: ``"docs"``).
        dimensions: Dimensionality of embedding vectors (default: ``384``).
    """

    db_path: str
    table_name: str = "docs"
    dimensions: int = 384


class VectorStore:
    """A sqlite-vec backed vector store for cosine-similarity search.

    Wraps a single :class:`sqlite3.Connection` and a vec0 virtual table.
    Supports atomic batch insertion and nearest-neighbour queries.

    Args:
        config: VectorStoreConfig with database path, table name, and dimensions.

    Example:
        >>> store = VectorStore(VectorStoreConfig(db_path="kb.db"))
        >>> store.initialize()
        >>> store.insert(["doc text"], [[0.1, 0.2, ...]])
        >>> results = store.search([0.3, 0.4, ...], limit=5)
        >>> results[0]["text"]
        'doc text'
    """

    def __init__(self, config: VectorStoreConfig) -> None:
        """Initialize the store with a configuration.

        The database connection is opened lazily during :meth:`initialize`.
        """
        self._config = config
        self._conn: sqlite3.Connection | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Open the database and create the vec0 table if it does not exist.

        Idempotent: calling ``initialize()`` multiple times is safe.
        """
        conn = sqlite3.connect(self._config.db_path)
        conn.enable_load_extension(True)
        conn.load_extension(sqlite_vec.loadable_path())

        conn.execute(
            f"""CREATE VIRTUAL TABLE IF NOT EXISTS {self._config.table_name}
            USING vec0(embedding float[{self._config.dimensions}], text TEXT)"""
        )
        conn.commit()
        self._conn = conn

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def insert(self, texts: List[str], embeddings: List[List[float]]) -> None:
        """Insert a batch of texts with their corresponding embedding vectors.

        Args:
            texts: Human-readable strings to store alongside embeddings.
            embeddings: One 384-dim float vector per text. Must have the same
                length as *texts*.

        Raises:
            ValueError: If ``len(texts) != len(embeddings)``.
        """
        if len(texts) != len(embeddings):
            raise ValueError(
                f"Mismatch: {len(texts)} texts vs {len(embeddings)} embeddings"
            )

        conn = self._get_conn()
        with conn:
            for text, embedding in zip(texts, embeddings):
                serialized = sqlite_vec.serialize_float32(embedding)
                conn.execute(
                    f"INSERT INTO {self._config.table_name} (text, embedding) "
                    "VALUES (?, ?)",
                    (text, serialized),
                )

    def search(
        self, query_embedding: List[float], limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Search for the *limit* nearest neighbours by cosine similarity.

        Args:
            query_embedding: A 384-dim float vector to compare against.
            limit: Maximum number of results to return (default: ``5``).

        Returns:
            A list of dicts with keys ``"text"`` (str) and ``"score"`` (float).
            Results are ordered by descending similarity (least distance first).
            Returns an empty list when the store has no rows.
        """
        conn = self._get_conn()
        serialized = sqlite_vec.serialize_float32(query_embedding)

        rows = conn.execute(
            f"""SELECT text, distance
            FROM {self._config.table_name}
            WHERE embedding MATCH ?
            ORDER BY distance
            LIMIT ?""",
            (serialized, limit),
        ).fetchall()

        return [{"text": row[0], "score": row[1]} for row in rows]

    # ------------------------------------------------------------------
    # KnowledgeProtocol (Agno integration)
    # ------------------------------------------------------------------

    def build_context(self, **kwargs: Any) -> str:
        """Return instructions on how to use this knowledge base."""
        return (
            "Use semantic_search(query, limit) to find relevant Agno "
            "documentation chunks from the local knowledge base."
        )

    def get_tools(self, **kwargs: Any) -> List[Callable[..., Any]]:
        """Return synchronous search tools for the agent."""
        return [self._semantic_search_tool]

    async def aget_tools(self, **kwargs: Any) -> List[Callable[..., Any]]:
        """Async version of get_tools."""
        return self.get_tools(**kwargs)

    def retrieve(self, query: str, **kwargs: Any) -> List[Document]:
        """Retrieve relevant documents for context injection."""
        import asyncio as _async

        return _async.get_event_loop().run_until_complete(
            self.aretrieve(query, **kwargs)
        )

    async def aretrieve(
        self, query: str, max_results: int = 5, **kwargs: Any
    ) -> List[Document]:
        """Async retrieval: embed query, search, return Documents."""
        from agno_docs_agent.knowledge.embedder import EmbedderConfig, LocalEmbedder

        embedder = LocalEmbedder(EmbedderConfig())
        query_emb = await asyncio.to_thread(embedder.embed, query)
        results = await asyncio.to_thread(self.search, query_emb, limit=max_results)
        return [
            Document(content=r["text"], meta_data={"score": r["score"]})
            for r in results
        ]

    # ------------------------------------------------------------------
    # Tool helpers
    # ------------------------------------------------------------------

    def _semantic_search_tool(self, query: str, limit: int = 5) -> str:
        """Agent-callable semantic search. Embeds query and returns top chunks."""
        from agno_docs_agent.knowledge.embedder import EmbedderConfig, LocalEmbedder

        embedder = LocalEmbedder(EmbedderConfig())
        query_embedding = embedder.embed(query)
        results = self.search(query_embedding, limit=limit)

        if not results:
            return "No matching documents found."

        lines = [f"Semantic search results for: {query}"]
        for i, r in enumerate(results, 1):
            lines.append(f"\n--- Result {i} (score: {r['score']:.4f}) ---")
            lines.append(r["text"][:1000])
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_conn(self) -> sqlite3.Connection:
        """Return the database connection, raising if not yet initialized."""
        if self._conn is None:
            raise RuntimeError("VectorStore not initialized — call initialize() first")
        return self._conn
