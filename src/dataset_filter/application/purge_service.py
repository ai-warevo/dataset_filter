"""Application layer services coordinating domain workflows."""

from __future__ import annotations
from pathlib import Path
from tqdm import tqdm

from dataset_filter.domain.exceptions import ImageProcessingError
from dataset_filter.domain.models import FilterConfig, MediaFile
from dataset_filter.domain.interfaces import Detector, DefectStrategy, MatchStrategy


class DatasetPurgeService:
    """Application Service coordinating the core data filtering lifecycle workflows."""

    def __init__(
        self, 
        detectors: list[Detector], 
        config: FilterConfig, 
        defect_strategy: DefectStrategy,
        match_strategy: MatchStrategy | None = None
    ) -> None:
        self._detectors: list[Detector] = detectors
        self._config: FilterConfig = config
        self._defect_strategy: DefectStrategy = defect_strategy
        self._match_strategy: MatchStrategy | None = match_strategy

    def process_directory(self, target_directory: Path) -> None:
        """Scan directory boundaries, sorting real assets from structural anomalies."""
        all_targets: list[MediaFile] = [
            MediaFile(f)
            for f in target_directory.iterdir()
            if f.is_file() and f.suffix.lower() in self._config.valid_extensions
        ]

        tqdm.write(f"🔍 Found {len(all_targets)} images to parse. Starting service...")
        stats: dict[str, int] = {"passed": 0, "defects": 0}

        with tqdm(all_targets, desc="Filtering dataset", unit="img") as pbar:
            for item in pbar:
                self._inspect_item(item, stats)

        tqdm.write("\n📊 Purge workflow finalized!")
        tqdm.write(f"   Defects handled: {stats['defects']} files.")
        tqdm.write(f"   Retained items: {stats['passed']} files within '{target_directory}'.")

    def _inspect_item(self, item: MediaFile, stats: dict[str, int]) -> None:
        try:
            last_score = None
            for detector in self._detectors:
                result = detector.check(item.path)
                if result.is_defect:
                    # A defect was found by one of the detectors
                    self._defect_strategy.handle(item)
                    stats["defects"] += 1
                    return
                
                if result.score is not None:
                    last_score = result.score

            # If we reached here, it passed all detectors
            stats["passed"] += 1
            if self._match_strategy:
                self._match_strategy.handle(item, last_score)
            return

        except ImageProcessingError as err:
            tqdm.write(f"💥 Defect detected (System exception: {err}) #{item.path}")
            # If we can't even process the image, it's a defect
            self._defect_strategy.handle(item)
            stats["defects"] += 1
