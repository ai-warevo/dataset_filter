"""Abstract interface for detecting defects in media files."""

from abc import ABC, abstractmethod
from pathlib import Path


class Detector(ABC):
    """Abstract interface for detecting defects in media files."""

    @abstractmethod
    def check(self, file_path: Path) -> "DetectionResult":
        """Perform the detection check. Returns a DetectionResult."""
