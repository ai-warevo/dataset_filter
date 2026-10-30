from __future__ import annotations
from abc import ABC, abstractmethod
from dataset_filter.domain.models.embedding import FaceEmbedding


class IEmbeddingRepository(ABC):
    """Interface for storing and querying facial embeddings to detect duplicates."""

    @abstractmethod
    def add(self, embedding: FaceEmbedding) -> None:
        """Store a new unique embedding."""

    @abstractmethod
    def is_duplicate(self, embedding: FaceEmbedding, threshold: float) -> bool:
        """Check if an embedding is too similar to any existing one in the repository."""
        pass
