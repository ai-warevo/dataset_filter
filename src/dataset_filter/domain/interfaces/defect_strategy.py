"""Abstract strategy for handling items that fail validation or processing."""

from abc import ABC, abstractmethod
from dataset_filter.domain.models import MediaFile


class DefectStrategy(ABC):
    @abstractmethod
    def handle(self, item: MediaFile, reason: str = "") -> None:
        """Execute the defect handling logic (e.g., purge, log, sort)."""
