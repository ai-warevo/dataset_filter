from __future__ import annotations
import numpy as np
from dataset_filter.domain.interfaces.embedding_repository import IEmbeddingRepository
from dataset_filter.domain.models.embedding import FaceEmbedding


class InMemoryEmbeddingRepository(IEmbeddingRepository):
    """In-memory implementation of embedding repository using NumPy for fast distance calculation."""

    def __init__(self) -> None:
        self._embeddings: list[np.ndarray] = []

    def add(self, embedding: FaceEmbedding) -> None:
        self._embeddings.append(np.array(embedding.vector))

    def is_duplicate(self, embedding: FaceEmbedding, threshold: float) -> bool:
        if not self._embeddings:
            return False

        new_vec = np.array(embedding.vector)
        # Calculate Euclidean distances to all stored embeddings
        distances = np.linalg.norm(self._embeddings - new_vec, axis=1)
        
        return bool(np.any(distances < threshold))
