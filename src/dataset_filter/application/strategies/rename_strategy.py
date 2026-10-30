from __future__ import annotations
"""Universal strategy for renaming both defective and matching files."""

import re
from tqdm import tqdm
from dataset_filter.domain.models import MediaFile
from dataset_filter.domain.interfaces import DefectStrategy, MatchStrategy


class RenameStrategy(DefectStrategy, MatchStrategy):
    """
    Renames files using a suffix-based approach: [sanitized_name]__[status][score].ext
    This allows grouping by subject and easy comparison of scores.
    Example: 'frontal_chest_support_001__m28.png' or 'comfyui_001__d.png'
    """

    def _sanitize_name(self, stem: str) -> str:
        """Cleans the filename stem: lowercase, removes leading junk digits, and trailing underscores."""
        # 1. Lowercase
        clean = stem.lower()

        # 2. Remove leading digits that look like noise (e.g., '7comfyui' -> 'comfyui')
        # We use a lookahead to ensure we don't strip the first letter
        clean = re.sub(r"^\d+(?=[a-z])", "", clean)

        # 3. Remove trailing underscores and extra whitespace
        clean = clean.strip("_").strip()

        # 4. Collapse multiple underscores into one
        clean = re.sub(r"_+", "_", clean)

        return clean

    def handle(self, item: MediaFile, score: float | None = None) -> None:
        current_name = item.path.name

        # Check if already processed by this strategy to avoid double renaming
        if "__m" in current_name or "__d" in current_name:
            return

        try:
            stem = item.path.stem
            suffix = item.path.suffix
            clean_stem = self._sanitize_name(stem)

            if score is not None:
                # Match logic: [subject]__m[score].ext
                score_val = int(round(score * 100))
                new_name = f"{clean_stem}__m{score_val}{suffix}"
            else:
                # Defect logic: [subject]__d.ext
                new_name = f"{clean_stem}__d{suffix}"

            new_path = item.path.with_name(new_name)
            item.path.rename(new_path)

        except Exception as e:
            tqdm.write(f"❌ Failed to rename {current_name}: {e}")
