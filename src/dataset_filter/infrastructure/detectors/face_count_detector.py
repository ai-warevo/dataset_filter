from __future__ import annotations
from pathlib import Path
import face_recognition
from dataset_filter.domain.interfaces import Detector
from dataset_filter.domain.models import FilterConfig
from dataset_filter.domain.models.detection_result import DetectionResult

class FaceCountDetector(Detector):
    """Detects if there are too many faces in the image."""

    def __init__(self, config: FilterConfig) -> None:
        self._config = config

    def check(self, file_path: Path) -> DetectionResult:
        try:
            image = face_recognition.load_image_file(file_path)
            face_locations = face_recognition.face_locations(image)
            count = len(face_locations)

            if count == 0:
                return DetectionResult(is_defect=True, confidence=0.0, reason="No faces detected")

            is_defect = count > self._config.quality.face_count_max
            # Confidence decreases as the number of faces exceeds the limit
            confidence = 1.0 if not is_defect else (self._config.quality.face_count_max / count)
            reason = "" if not is_defect else f"Too many faces ({count})"

            return DetectionResult(is_defect=is_defect, confidence=confidence, reason=reason)
        except Exception as e:
            return DetectionResult(is_defect=True, confidence=0.0, reason=f"Face count error: {str(e)}")

    def get_weight(self) -> float:
        """Return the weight of this detector as defined in the configuration."""
        return self._config.quality.face_count_weight
