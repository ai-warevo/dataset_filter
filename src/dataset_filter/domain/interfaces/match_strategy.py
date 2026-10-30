from __future__ import annotations
"""Abstract strategy for handling items that pass validation."""

from abc import ABC, abstractmethod


class MatchStrategy(ABC):
    @abstractmethod
    def handle(self, item: "MediaFile", score: float | None = None) -> None:
        """Execute the match handling logic (e.g., rename, move, keep)."""
