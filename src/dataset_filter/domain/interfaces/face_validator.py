from __future__ import annotations
"""Abstract interface defining the contract for facial authentication engines."""

from abc import ABC, abstractmethod
from pathlib import Path
from dataset_filter.domain.interfaces.detector import Detector
from dataset_filter.domain.models.detection_result import DetectionResult
from dataset_filter.domain.models.embedding import FaceEmbedding


class FaceValidator(Detector):
    @abstractmethod
    def get_score(self, file_path: Path) -> float:
        """Determine the similarity score (distance) of the target image. 
        Lower is better/closer match. Returns float('inf') if no face found.
        """

    @abstractmethod
    def get_embedding(self, file_path: Path) -> FaceEmbedding | None:
        """Extract facial embedding from the given image."""

    def check(self, file_path: Path) -> DetectionResult:
        """Default implementation of detector check using get_score."""
        # This will be overridden or used by implementations that know their tolerance.
        # For now, we'll let the concrete implementation decide how to use the score.
        raise NotImplementedError("Implementations must provide a way to determine if a score constitutes a defect.")
