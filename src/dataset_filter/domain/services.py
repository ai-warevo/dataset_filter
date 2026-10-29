"""Domain service interfaces."""

from abc import ABC, abstractmethod
from pathlib import Path


class FaceValidator(ABC):
    """Abstract interface defining the contract for facial authentication engines."""

    @abstractmethod
    def get_score(self, file_path: Path) -> float:
        """Determine the similarity score (distance) of the target image. 
        Lower is better/closer match. Returns float('inf') if no face found.
        """


class DefectStrategy(ABC):
    """Abstract strategy for handling items that fail validation or processing."""

    @abstractmethod
    def handle(self, item: "MediaFile") -> None:
        """Execute the defect handling logic (e.g., purge, log, sort)."""


class MatchStrategy(ABC):
    """Abstract strategy for handling items that pass validation."""

    @abstractmethod
    def handle(self, item: "MediaFile", score: float) -> None:
        """Execute the match handling logic (e.g., rename, move, keep)."""
