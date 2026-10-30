"""Abstract strategy for handling items that fail validation or processing."""

from abc import ABC, abstractmethod


class DefectStrategy(ABC):
    @abstractmethod
    def handle(self, item: "MediaFile") -> None:
        """Execute the defect handling logic (e.g., purge, log, sort)."""
