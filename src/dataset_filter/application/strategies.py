"""Concrete implementations of defect and match handling strategies."""

import re
from tqdm import tqdm
from dataset_filter.domain.services import DefectStrategy, MatchStrategy
from dataset_filter.domain.models import MediaFile


class PurgeDefectStrategy(DefectStrategy):
    """Strategy that permanently deletes the defective file."""

    def handle(self, item: MediaFile) -> None:
        item.purge()


class LogDefectStrategy(DefectStrategy):
    """Strategy that only logs the presence of a defect without modifying the filesystem."""

    def handle(self, item: MediaFile) -> None:
        tqdm.write(f"⚠️  Logged defect: {item.path}")


class NoOpMatchStrategy(MatchStrategy):
    """Strategy that does nothing with matching files."""

    def handle(self, item: MediaFile, score: float) -> None:
        pass


class RenameMatchStrategy(MatchStrategy):
    """Strategy that renames matching files by adding their similarity score as a prefix."""

    # Pattern to detect if the file already has a match prefix like 'match55_' or 'match0.55_'
    _PREFIX_PATTERN = re.compile(r"^match[\d\.]+_")

    def handle(self, item: MediaFile, score: float) -> None:
        if self._PREFIX_PATTERN.match(item.path.name):
            return

        # Convert score to a readable format (e.g., 0.55 -> 55)
        score_val = int(round(score * 100))
        new_name = f"match{score_val}_{item.path.name}"
        new_path = item.path.with_name(new_name)
        
        try:
            item.path.rename(new_path)
        except Exception as e:
            tqdm.write(f"❌ Failed to rename {item.path.name} -> {new_name}: {e}")


class RenameDefectStrategy(DefectStrategy):
    """Strategy that renames defective files by adding a 'defect_' prefix."""

    def handle(self, item: MediaFile) -> None:
        if item.path.name.startswith("defect_"):
            return

        new_name = f"defect_{item.path.name}"
        new_path = item.path.with_name(new_name)
        
        try:
            item.path.rename(new_path)
        except Exception as e:
            tqdm.write(f"❌ Failed to rename defect {item.path.name} -> {new_name}: {e}")
