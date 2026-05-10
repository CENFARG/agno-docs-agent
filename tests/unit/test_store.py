"""Unit tests for SqliteVecDb — Agno VectorDb ABC implementation backed by sqlite-vec.

Tests create, insert, search, exists, drop, delete, and ABC compliance.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from agno.knowledge.document.base import Document

from agno_docs_agent.knowledge.store import SqliteVecDb, SqliteVecDbConfig


class TestSqliteVecDb:
    """Verify SqliteVecDb conforms to the VectorDb ABC and performs semantic search."""

    @pytest.fixture
    def config(self) -> SqliteVecDbConfig:
        """Return a config pointing to a fresh temporary database."""
        tmpdir = tempfile.mkdtemp()
        db_path = Path(tmpdir) / "test_knowledge.db"
        return SqliteVecDbConfig(db_path=str(db_path), dimensions=384)

    @pytest.fixture
    def embedder(self) -> object:
        """Return a stub embedder that returns deterministic embeddings."""
        return _FakeEmbedder()

    # ------------------------------------------------------------------
    # create / exists / drop / delete
    # ------------------------------------------------------------------

    def test_create_initializes_database(self, config: SqliteVecDbConfig, embedder) -> None:
        """create() MUST initialise the sqlite-vec table."""
        db = SqliteVecDb(config=config, embedder=embedder)
        db.create()

        assert db.exists()

    def test_exists_returns_false_before_create(
        self, config: SqliteVecDbConfig, embedder
    ) -> None:
        """exists() MUST return False before create() is called."""
        db = SqliteVecDb(config=config, embedder=embedder)
        assert db.exists() is False

    def test_exists_returns_true_after_create(
        self, config: SqliteVecDbConfig, embedder
    ) -> None:
        """exists() MUST return True after create() completes."""
        db = SqliteVecDb(config=config, embedder=embedder)
        db.create()
        assert db.exists() is True

    def test_drop_destroys_table(self, config: SqliteVecDbConfig, embedder) -> None:
        """drop() MUST destroy the vector table."""
        db = SqliteVecDb(config=config, embedder=embedder)
        db.create()
        assert db.exists()

        db.drop()
        assert not db.exists()

    def test_create_idempotent(self, config: SqliteVecDbConfig, embedder) -> None:
        """Calling create() multiple times MUST be safe."""
        db = SqliteVecDb(config=config, embedder=embedder)
        db.create()
        db.create()  # should not raise
        assert db.exists()

    def test_delete_removes_underlying_storage(
        self, config: SqliteVecDbConfig, embedder
    ) -> None:
        """delete() MUST return True and make exists() return False."""
        db = SqliteVecDb(config=config, embedder=embedder)
        db.create()
        assert db.exists()

        result = db.delete()
        assert result is True
        assert not db.exists()

    # ------------------------------------------------------------------
    # insert / search
    # ------------------------------------------------------------------

    def test_insert_and_search_returns_documents(
        self, config: SqliteVecDbConfig, embedder
    ) -> None:
        """insert() + search() MUST return ranked Document objects."""
        db = SqliteVecDb(config=config, embedder=embedder)
        db.create()

        docs = [
            Document(content="python is a programming language"),
            Document(content="agno is a framework for building AI agents"),
        ]
        db.insert(content_hash="hash1", documents=docs)

        results = db.search(query="AI agents framework", limit=2)

        assert len(results) == 2
        assert isinstance(results[0], Document)
        assert "agno" in results[0].content.lower()
        assert isinstance(results[1], Document)

    def test_search_returns_empty_for_empty_store(
        self, config: SqliteVecDbConfig, embedder
    ) -> None:
        """search() on an empty store MUST return an empty list."""
        db = SqliteVecDb(config=config, embedder=embedder)
        db.create()

        results = db.search(query="anything", limit=10)
        assert results == []

    def test_insert_respects_content_hash(
        self, config: SqliteVecDbConfig, embedder
    ) -> None:
        """content_hash_exists() MUST return True after insert with that hash."""
        db = SqliteVecDb(config=config, embedder=embedder)
        db.create()

        docs = [Document(content="unique content for hashing")]
        db.insert(content_hash="abc123", documents=docs)

        assert db.content_hash_exists("abc123") is True
        assert db.content_hash_exists("nonexistent") is False

    def test_search_limit_respected(
        self, config: SqliteVecDbConfig, embedder
    ) -> None:
        """search() MUST respect the *limit* parameter."""
        db = SqliteVecDb(config=config, embedder=embedder)
        db.create()

        docs = [Document(content=f"document {i}") for i in range(10)]
        db.insert(content_hash="batch", documents=docs)

        results = db.search(query="document", limit=3)
        assert len(results) == 3

    # ------------------------------------------------------------------
    # ABC compliance: minimals that MUST work
    # ------------------------------------------------------------------

    def test_is_vector_db_subclass(self) -> None:
        """SqliteVecDb MUST be a subclass of agno.vectordb.base.VectorDb."""
        from agno.vectordb.base import VectorDb

        assert issubclass(SqliteVecDb, VectorDb)

    def test_name_exists_default_implementation(
        self, config: SqliteVecDbConfig, embedder
    ) -> None:
        """name_exists() MUST have a working default (even if NotImplemented)."""
        db = SqliteVecDb(config=config, embedder=embedder)
        # Should not raise AttributeError (method exists)
        assert callable(db.name_exists)


# ------------------------------------------------------------------
# Fake embedder for deterministic tests (no real model load)
# ------------------------------------------------------------------


class _FakeEmbedder:
    """Deterministic embedder that returns predictable embeddings.

    Uses a simple hash-based approach so that semantically different
    texts produce different vectors, and similar prefixes yield
    similar vectors (enough for the cosine-similarity vector search
    to return meaningful rankings in tests).
    """

    dimensions: int = 384

    def get_embedding(self, text: str) -> list[float]:
        """Return a deterministic 384-dim embedding for *text*."""
        if isinstance(text, list):
            return [self.get_embedding(t) for t in text]
        # Simple hash-bashed embedding: each character contributes to a dimension
        result = [0.0] * self.dimensions
        for i, ch in enumerate(text.lower()):
            idx = (hash(ch) * (i + 1)) % self.dimensions
            result[idx] += 0.1
        # Normalize
        norm = (sum(v * v for v in result)) ** 0.5 or 1.0
        return [v / norm for v in result]

    def get_embedding_and_usage(self, text: str) -> tuple[list[float], None]:
        """Return embedding and None usage."""
        return self.get_embedding(text), None

    async def async_get_embedding(self, text: str) -> list[float]:
        """Async version of get_embedding."""
        return self.get_embedding(text)

    async def async_get_embedding_and_usage(self, text: str) -> tuple[list[float], None]:
        """Async version of get_embedding_and_usage."""
        return self.get_embedding_and_usage(text)
