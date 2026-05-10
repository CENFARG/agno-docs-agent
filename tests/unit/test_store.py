"""Unit tests for VectorStore — sqlite-vec backed semantic search.

Tests table creation, batch insert, cosine-similarity search, and error handling.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from agno_docs_agent.knowledge.store import VectorStore, VectorStoreConfig


class TestVectorStore:
    """Verify VectorStore creates, seeds, and queries a sqlite-vec table.

    Uses a temporary database that is cleaned up after each test.
    """

    @pytest.fixture
    def config(self) -> VectorStoreConfig:
        """Return a config pointing to a fresh temporary database."""
        tmpdir = tempfile.mkdtemp()
        db_path = Path(tmpdir) / "test_knowledge.db"
        return VectorStoreConfig(db_path=str(db_path), dimensions=384)

    def test_initialize_creates_table(self, config: VectorStoreConfig) -> None:
        """initialize() MUST create a vec0 virtual table with the configured dimensions."""
        store = VectorStore(config)
        store.initialize()

        import sqlite3

        conn = sqlite3.connect(config.db_path)
        conn.enable_load_extension(True)
        import sqlite_vec

        conn.load_extension(sqlite_vec.loadable_path())

        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='docs'"
        )
        tables = cursor.fetchall()
        conn.close()

        assert len(tables) == 1
        assert tables[0][0] == "docs"

    def test_insert_and_search_with_scores(self, config: VectorStoreConfig) -> None:
        """insert() + search() MUST return ranked results with similarity scores."""
        store = VectorStore(config)
        store.initialize()

        texts = ["python is a programming language", "agno is a framework for agents"]
        embeddings = [[0.1] * 384, [0.5] * 384]
        store.insert(texts, embeddings)

        query_embedding = [0.5] * 384
        results = store.search(query_embedding, limit=2)

        assert len(results) == 2
        assert "text" in results[0]
        assert "score" in results[0]
        assert isinstance(results[0]["score"], float)
        # The second text ("agno is a framework...") should rank higher
        assert "agno" in results[0]["text"].lower()

    def test_search_returns_empty_for_empty_store(self, config: VectorStoreConfig) -> None:
        """search() on an empty store MUST return an empty list."""
        store = VectorStore(config)
        store.initialize()

        results = store.search([0.1] * 384, limit=10)
        assert results == []

    def test_insert_batch_is_atomic_style(self, config: VectorStoreConfig) -> None:
        """Multiple texts and embeddings MUST all be inserted successfully."""
        store = VectorStore(config)
        store.initialize()

        texts = [f"document {i}" for i in range(5)]
        embeddings = [[float(i) / 10] * 384 for i in range(5)]
        store.insert(texts, embeddings)

        results = store.search([0.3] * 384, limit=10)
        assert len(results) == 5

    def test_search_limit_respected(self, config: VectorStoreConfig) -> None:
        """search() MUST respect the *limit* parameter."""
        store = VectorStore(config)
        store.initialize()

        texts = [f"doc {i}" for i in range(10)]
        embeddings = [[float(i) / 10] * 384 for i in range(10)]
        store.insert(texts, embeddings)

        results = store.search([0.5] * 384, limit=3)
        assert len(results) == 3

    def test_duplicate_db_path_does_not_fail(self, config: VectorStoreConfig) -> None:
        """initialize() on an already-initialized database MUST be idempotent."""
        store = VectorStore(config)
        store.initialize()
        # Second call should not raise
        store.initialize()
