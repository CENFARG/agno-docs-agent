"""Local embedding service wrapping sentence-transformers.

Provides a lazy-loaded, cached SentenceTransformer model for computing text embeddings.
All vectors are returned as plain Python float lists of fixed dimension.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from sentence_transformers import SentenceTransformer


class ModelNotFound(Exception):
    """Raised when the configured SentenceTransformer model cannot be loaded."""


@dataclass(frozen=True)
class EmbedderConfig:
    """Configuration for the local embedding model.

    Attributes:
        model_name: HuggingFace model identifier.
            Defaults to ``all-MiniLM-L6-v2`` (384-dim embeddings).
    """

    model_name: str = "all-MiniLM-L6-v2"


class LocalEmbedder:
    """Lazy-loading wrapper around a SentenceTransformer model.

    The model is loaded on first access and cached for the lifetime of the instance.
    Embeddings are returned as Python ``list[float]`` for downstream simplicity.

    Args:
        config: EmbedderConfig specifying the model name and options.

    Raises:
        ModelNotFound: If the configured model cannot be loaded.
    """

    def __init__(self, config: EmbedderConfig) -> None:
        """Initialize the embedder with a configuration."""
        self._config = config
        self._model: Optional[SentenceTransformer] = None

    @property
    def model(self) -> SentenceTransformer:
        """Return the cached SentenceTransformer model, loading it if necessary.

        Returns:
            The loaded SentenceTransformer instance.

        Raises:
            ModelNotFound: If the model cannot be loaded from HuggingFace hub or cache.
        """
        if self._model is None:
            try:
                self._model = SentenceTransformer(self._config.model_name)
            except Exception as exc:
                raise ModelNotFound(
                    f"Failed to load model '{self._config.model_name}': {exc}"
                ) from exc
        return self._model

    def embed(self, text: str) -> List[float]:
        """Compute a single embedding vector for *text*.

        Args:
            text: Input string to embed.

        Returns:
            A list of floats whose length equals the model's embedding dimension
            (384 for the default ``all-MiniLM-L6-v2``).

        Raises:
            ModelNotFound: If the underlying model cannot be loaded.
        """
        embedding = self.model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        result: List[float] = embedding.tolist()
        return result

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Compute embeddings for a batch of texts.

        Args:
            texts: List of strings to embed. May be empty.

        Returns:
            A list of embedding vectors, one per input text. Returns an empty list
            when *texts* is empty.

        Raises:
            ModelNotFound: If the underlying model cannot be loaded.
        """
        if not texts:
            return []
        embeddings = self.model.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True
        )
        result: List[List[float]] = embeddings.tolist()
        return result
