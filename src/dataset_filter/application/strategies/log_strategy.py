from __future__ import annotations
"""Universal strategy for logging both defects and matches."""

from tqdm import tqdm
from dataset_filter.domain.models import MediaFile
from dataset_filter.domain.interfaces import DefectStrategy, MatchStrategy


class LogStrategy(DefectStrategy, MatchStrategy):
    def handle(self, item: MediaFile, score: float | None = None, reason: str = "") -> None:
        if score is not None:
            tqdm.write(f"✅ Matched: {item.path.name} (score: {score:.2f})")
        elif reason:
            tqdm.write(f"⚠️  Logged defect: {item.path.name} ({reason})")
        else:
            tqdm.write(f"⚠️  Logged defect: {item.path.name}")
