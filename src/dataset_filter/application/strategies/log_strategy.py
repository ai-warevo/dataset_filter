from __future__ import annotations
"""Universal strategy for logging both defects and matches."""

import re
from tqdm import tqdm
from dataset_filter.domain.models import MediaFile
from dataset_filter.domain.interfaces import DefectStrategy, MatchStrategy


class LogStrategy(DefectStrategy, MatchStrategy):
    def handle(self, item: MediaFile, score: float | None = None) -> None:
        if score is not None:
            tqdm.write(f"✅ Matched: {item.path.name} (score: {score:.2f})")
        else:
            tqdm.write(f"⚠️  Logged defect: {item.path.name}")
