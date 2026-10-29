"""Domain value objects and entities."""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FilterConfig:
    """Value Object encapsulating threshold parameters and strict type policies."""

    tolerance: float = 0.55
    valid_extensions: tuple[str, ...] = (".png", ".jpg", ".jpeg")


class MediaFile:
    """DDD Entity representing an individual file artifact inside the dataset."""

    def __init__(self, path: Path) -> None:
        self.path: Path = path

    def purge(self) -> None:
        """Atomically delete the underlying infrastructure file resource."""
        self.path.unlink(missing_ok=True)
