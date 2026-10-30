from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence

@dataclass(frozen=True)
class FaceEmbedding:
    """Domain Value Object representing a facial embedding vector."""
    vector: Sequence[float]

    def __len__(self) -> int:
        return len(self.vector)
