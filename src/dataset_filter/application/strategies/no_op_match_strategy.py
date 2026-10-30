from __future__ import annotations
"""Strategy that does nothing with matching files."""

from dataset_filter.domain.models import MediaFile
from dataset_filter.domain.interfaces import MatchStrategy


class NoOpMatchStrategy(MatchStrategy):
    def handle(self, item: MediaFile, score: float | None = None) -> None:
        pass
