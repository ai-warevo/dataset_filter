"""Application layer services coordinating the multi-stage pipeline workflow."""

from __future__ import annotations
from pathlib import Path
from typing import Collection

from dataset_filter.domain.interfaces.pipeline_stage import PipelineStage
from dataset_filter.domain.models.media_file import MediaFile


class PipelineOrchestrator:
    """Orchestrates the execution of multiple pipeline stages in sequence."""

    def __init__(self, stages: list[PipelineStage]) -> None:
        self._stages = stages

    def run(self, initial_assets: Collection[MediaFile]) -> Collection[MediaFile]:
        """Executes all configured stages sequentially on the provided assets."""
        current_assets = initial_assets
        for stage in self._stages:
            current_assets = stage.execute(current_assets)
        return current_assets

    def process_directory(self, target_directory: Path, valid_extensions: tuple[str, ...]) -> None:
        """High-level entry point that scans a directory and runs the pipeline."""
        # Note: In a real DDD implementation, scanning might be its own service or part of an infrastructure layer.
        # For this refactor, we keep it here to maintain compatibility with main.py's current flow.
        all_targets = [
            MediaFile(f)
            for f in target_directory.iterdir()
            if f.is_file() and f.suffix.lower() in valid_extensions
        ]

        print(f"🔍 Found {len(all_targets)} images to parse. Starting pipeline...")
        
        final_assets = self.run(all_targets)

        print("\n📊 Pipeline workflow finalized!")
        print(f"   Retained items: {len(final_assets)} files within '{target_directory}'.")
