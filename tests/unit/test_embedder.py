"""Unit tests for Agno's SentenceTransformerEmbedder.

Tests embedding single/batch texts, dimensions, and error handling
using the native Agno embedder from `agno.knowledge.embedder.sentence_transformer`.
"""

from __future__ import annotations

import numpy as np
import pytest
from agno.knowledge.embedder.sentence_transformer import SentenceTransformerEmbedder


class TestSentenceTransformerEmbedder:
    """Verify Agno's SentenceTransformerEmbedder works with the all-MiniLM-L6-v2 model."""

    @pytest.fixture
    def embedder(self) -> SentenceTransformerEmbedder:
        """Default embedder with the all-MiniLM-L6-v2 model (384-dim)."""
        return SentenceTransformerEmbedder(id="all-MiniLM-L6-v2", dimensions=384)

    def test_get_embedding_returns_correct_dimension(
        self, embedder: SentenceTransformerEmbedder
    ) -> None:
        """get_embedding() MUST return a 384-dim float vector for the default model."""
        vector = embedder.get_embedding("hello world")

        assert isinstance(vector, list)
        assert all(isinstance(v, float) for v in vector)
        assert len(vector) == 384

    def test_get_embedding_batch_returns_multiple_vectors(
        self, embedder: SentenceTransformerEmbedder
    ) -> None:
        """get_embedding() with a list of texts MUST return a list of 384-dim vectors."""
        texts = ["first sentence", "second sentence", "third sentence"]
        result = embedder.get_embedding(texts)

        assert isinstance(result, list)
        assert len(result) == 3
        for vec in result:
            assert isinstance(vec, list)
            assert len(vec) == 384

    def test_get_embedding_empty_input(self, embedder: SentenceTransformerEmbedder) -> None:
        """get_embedding() with an empty list SHOULD return an empty list."""
        result = embedder.get_embedding([])

        assert result == []

    def test_embedder_id_property(self, embedder: SentenceTransformerEmbedder) -> None:
        """The embedder MUST expose the model id as a dataclass field."""
        assert embedder.id == "all-MiniLM-L6-v2"
        assert embedder.dimensions == 384

    def test_get_embedding_returns_float_list_not_numpy(
        self, embedder: SentenceTransformerEmbedder
    ) -> None:
        """get_embedding() MUST return Python float list, not numpy array."""
        vector = embedder.get_embedding("test")

        assert isinstance(vector, list)
        assert not isinstance(vector, np.ndarray)

    def test_missing_model_raises_error(self) -> None:
        """A non-existent model name MUST raise an error at init time."""
        with pytest.raises(Exception):
            SentenceTransformerEmbedder(id="nonexistent-model-v99999")
