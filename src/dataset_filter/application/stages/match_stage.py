"""Implementation of the match handling stage in the pipeline."""

from __future__ import annotations
from typing import Collection

from tqdm import tqdm

from dataset_filter.domain.interfaces.pipeline_stage import PipelineStage
from dataset_filter.domain.interfaces import MatchStrategy
from dataset_filter.domain.models import MediaFile


class MatchStage(PipelineStage):
    """Stage that applies match strategies (e.g., renaming) to assets that passed all checks."""

    def __init__(self, match_strategy: MatchStrategy) -> None:
        self._match_strategy = match_strategy

    def execute(self, input_assets: Collection[MediaFile]) -> Collection[MediaFile]:
        """Applies the match strategy to each asset."""
        with tqdm(input_assets, desc="Applying matches", unit="img", leave=False) as pbar:
            for item in pbar:
                self._match_strategy.handle(item, score=item.score)
        
        return input_assets
