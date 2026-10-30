"""DDD Entity representing an individual file artifact inside the dataset."""

from pathlib import Path


class MediaFile:
    def __init__(self, path: Path) -> None:
        self.path: Path = path

    def purge(self) -> None:
        """Atomically delete the underlying infrastructure file resource."""
        self.path.unlink(missing_ok=True)
