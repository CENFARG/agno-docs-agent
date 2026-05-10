"""Unit tests for LocalEmbedder — wrapping sentence-transformers for semantic search.

Tests model loading, single/batch embedding, and error handling.
"""

from __future__ import annotations

import numpy as np
import pytest

from agno_docs_agent.knowledge.embedder import EmbedderConfig, LocalEmbedder, ModelNotFound


class TestLocalEmbedder:
    """Verify LocalEmbedder correctly wraps a SentenceTransformer model."""

    @pytest.fixture
    def config(self) -> EmbedderConfig:
        """Default configuration with a small reliable model."""
        return EmbedderConfig(model_name="all-MiniLM-L6-v2")

    def test_embed_returns_correct_dimension(self, config: EmbedderConfig) -> None:
        """embed() MUST return a 384-dim float vector for the default model."""
        embedder = LocalEmbedder(config)
        vector = embedder.embed("hello world")

        assert isinstance(vector, list)
        assert all(isinstance(v, float) for v in vector)
        assert len(vector) == 384

    def test_embed_batch_returns_multiple_vectors(self, config: EmbedderConfig) -> None:
        """embed_batch() MUST return one 384-dim vector per input text."""
        embedder = LocalEmbedder(config)
        texts = ["first sentence", "second sentence", "third sentence"]
        vectors = embedder.embed_batch(texts)

        assert isinstance(vectors, list)
        assert len(vectors) == 3
        for vec in vectors:
            assert isinstance(vec, list)
            assert len(vec) == 384

    def test_embed_batch_empty_input(self, config: EmbedderConfig) -> None:
        """embed_batch() with empty input SHOULD return an empty list."""
        embedder = LocalEmbedder(config)
        vectors = embedder.embed_batch([])

        assert vectors == []

    def test_model_reuses_instance(self, config: EmbedderConfig) -> None:
        """The underlying SentenceTransformer model MUST be loaded once and reused."""
        embedder = LocalEmbedder(config)
        model1 = embedder.model
        model2 = embedder.model

        assert model1 is model2

    def test_embed_returns_float_list_not_numpy(self, config: EmbedderConfig) -> None:
        """embed() MUST return Python float list, not numpy array."""
        embedder = LocalEmbedder(config)
        vector = embedder.embed("test")

        assert isinstance(vector, list)
        assert not isinstance(vector, np.ndarray)

    def test_missing_model_raises_custom_error(self) -> None:
        """A non-existent model name MUST raise ModelNotFound."""
        bad_config = EmbedderConfig(model_name="nonexistent-model-v99999")
        embedder = LocalEmbedder(bad_config)

        with pytest.raises(ModelNotFound):
            embedder.embed("anything")
