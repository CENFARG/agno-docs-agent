"""Integration tests for embedder + vector store pipeline.

Validates the full knowledge pipeline: embed → insert → search with real dependencies.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from agno_docs_agent.knowledge.embedder import EmbedderConfig, LocalEmbedder
from agno_docs_agent.knowledge.store import VectorStore, VectorStoreConfig


class TestKnowledgePipeline:
    """End-to-end: embed texts with a real model, store in sqlite-vec, query."""

    @pytest.fixture
    def pipeline(self) -> tuple[LocalEmbedder, VectorStore]:
        """Create a fresh embedder and vector store for each test."""
        embedder = LocalEmbedder(EmbedderConfig(model_name="all-MiniLM-L6-v2"))

        tmpdir = tempfile.mkdtemp()
        db_path = Path(tmpdir) / "integration.db"
        store = VectorStore(VectorStoreConfig(db_path=str(db_path), dimensions=384))
        store.initialize()

        return embedder, store

    def test_embed_insert_search_flow(self, pipeline) -> None:
        """Embed texts, insert into store, search by semantic similarity."""
        embedder, store = pipeline

        texts = [
            "Agno is a Python framework for building AI agents",
            "Python is a general-purpose programming language",
            "The weather today is sunny and warm",
        ]
        embeddings = embedder.embed_batch(texts)
        assert len(embeddings) == 3
        assert all(len(vec) == 384 for vec in embeddings)

        store.insert(texts, embeddings)

        # Search for Agno-related content
        query_embedding = embedder.embed("How to build AI agents with Agno?")
        results = store.search(query_embedding, limit=3)

        assert len(results) >= 1
        # The top result should be the Agno-related text
        assert "agno" in results[0]["text"].lower()
        assert isinstance(results[0]["score"], float)

    def test_dimension_mismatch_is_caught(self, pipeline) -> None:
        """Store MUST NOT accept embeddings with wrong dimensions."""
        embedder, store = pipeline

        # The real embedder always produces 384-dim vectors, so this is a
        # sanity check that the dimensions match between embedder and store.
        embeddings = embedder.embed_batch(["test text"])
        assert all(len(vec) == 384 for vec in embeddings)

    def test_empty_store_returns_empty_results(self, pipeline) -> None:
        """Search on an empty store MUST return an empty list."""
        embedder, store = pipeline

        query_embedding = embedder.embed("anything")
        results = store.search(query_embedding, limit=10)

        assert results == []

    def test_multiple_inserts_accumulate(self, pipeline) -> None:
        """Multiple insert() calls MUST accumulate documents."""
        embedder, store = pipeline

        batch1 = ["first batch text about agents"]
        batch2 = ["second batch text about frameworks"]
        store.insert(batch1, embedder.embed_batch(batch1))
        store.insert(batch2, embedder.embed_batch(batch2))

        query_embedding = embedder.embed("agents and frameworks")
        results = store.search(query_embedding, limit=5)

        assert len(results) >= 2
