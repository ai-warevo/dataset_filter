"""Infrastructure layer implementation of domain interfaces using face_recognition."""

from __future__ import annotations
from collections.abc import Sequence
from pathlib import Path
import face_recognition  # type: ignore[import-untyped]
from tqdm import tqdm

from dataset_filter.domain.exceptions import ImageProcessingError
from dataset_filter.domain.models import FilterConfig
from dataset_filter.domain.interfaces import FaceValidator
from dataset_filter.domain.models.detection_result import DetectionResult


class RecognitionEngine(FaceValidator):
    """Concrete implementation of FaceValidator leveraging dlib face_recognition."""

    def __init__(self, reference_path: Path, config: FilterConfig) -> None:
        self._config: FilterConfig = config
        self._ref_vector: Sequence[float] = self._extract_reference_vector(reference_path)

    def check(self, file_path: Path) -> DetectionResult:
        """Check if the face in the image is a defect based on tolerance."""
        try:
            score = self.get_score(file_path)
            return DetectionResult(is_defect=score > self._config.tolerance, score=score)
        except ImageProcessingError as err:
            # If we can't even process the image, it's a defect (or at least an error)
            return DetectionResult(is_defect=True)

    def get_score(self, file_path: Path) -> float:
        try:
            current_image = face_recognition.load_image_file(file_path)
            current_encodings = face_recognition.face_encodings(current_image)

            if not current_encodings:
                return float('inf')

            distances = face_recognition.face_distance(current_encodings, self._ref_vector)
            # Return the minimum distance found (best match)
            return min(distances)

        except Exception as err:  # noqa: BLE001
            raise ImageProcessingError(f"Corrupted or unsafe file structure: {err}") from err

    def _extract_reference_vector(self, path: Path) -> Sequence[float]:
        if not path.is_file():
            raise FileNotFoundError(f"Reference master asset '{path}' is missing.")

        try:
            ref_image = face_recognition.load_image_file(path)
            if encodings := face_recognition.face_encodings(ref_image):
                return encodings[0]
        except Exception as err:  # noqa: BLE001
            raise ImageProcessingError(f"Failed parsing reference master matrix: {err}") from err

        raise ValueError(f"Zero clear facial tracking signatures extracted from: {path}")
