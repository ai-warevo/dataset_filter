from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Collection

from dataset_filter.domain.models.media_file import MediaFile


class PipelineStage(ABC):
    """Domain interface for a single processing step in the dataset purification pipeline."""

    @abstractmethod
    def execute(self, input_assets: Collection[MediaFile]) -> Collection[MediaFile]:
        """Processes a collection of assets and returns the subset that should proceed to the next stage."""
        pass
