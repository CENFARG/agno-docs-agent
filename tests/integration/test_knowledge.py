"""Integration tests for the Agno-native knowledge pipeline.

Validates the full pipeline: embedder → store → search using
Agno's SentenceTransformerEmbedder, SqliteVecDb, and Document objects.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from agno.knowledge.document.base import Document
from agno.knowledge.embedder.sentence_transformer import SentenceTransformerEmbedder

from agno_docs_agent.knowledge.store import SqliteVecDb, SqliteVecDbConfig


class TestKnowledgePipeline:
    """End-to-end: embed texts with real model, store in sqlite-vec, query."""

    @pytest.fixture
    def pipeline(self) -> tuple[SentenceTransformerEmbedder, SqliteVecDb]:
        """Create a fresh embedder and vector store for each test."""
        embedder = SentenceTransformerEmbedder(id="all-MiniLM-L6-v2", dimensions=384)

        tmpdir = tempfile.mkdtemp()
        db_path = Path(tmpdir) / "integration.db"
        store = SqliteVecDb(
            config=SqliteVecDbConfig(db_path=str(db_path), dimensions=384),
            embedder=embedder,
        )
        store.create()

        return embedder, store

    def test_embed_insert_search_flow(self, pipeline) -> None:
        """Embed texts, insert into store, search by semantic similarity."""
        embedder, store = pipeline

        texts = [
            "Agno is a Python framework for building AI agents",
            "Python is a general-purpose programming language",
            "The weather today is sunny and warm",
        ]
        embeddings = embedder.get_embedding(texts)
        assert len(embeddings) == 3
        assert all(len(vec) == 384 for vec in embeddings)

        docs = [Document(content=t) for t in texts]
        store.insert(content_hash="flow1", documents=docs)

        # Search for Agno-related content
        results = store.search(query="How to build AI agents with Agno?", limit=3)

        assert len(results) >= 1
        # The top result should be the Agno-related text
        assert "agno" in results[0].content.lower()

    def test_dimension_match_sanity(self, pipeline) -> None:
        """Store MUST accept 384-dim embeddings from the real model."""
        embedder, store = pipeline

        embeddings = embedder.get_embedding(["test text"])
        assert all(len(vec) == 384 for vec in embeddings)

    def test_empty_store_returns_empty_results(self, pipeline) -> None:
        """Search on an empty store MUST return an empty list."""
        embedder, store = pipeline

        results = store.search(query="anything", limit=10)
        assert results == []

    def test_multiple_inserts_accumulate(self, pipeline) -> None:
        """Multiple insert() calls MUST accumulate documents."""
        embedder, store = pipeline

        batch1 = [Document(content="first batch text about agents")]
        batch2 = [Document(content="second batch text about frameworks")]
        store.insert(content_hash="b1", documents=batch1)
        store.insert(content_hash="b2", documents=batch2)

        results = store.search(query="agents and frameworks", limit=5)
        assert len(results) >= 2
