from __future__ import annotations
from pathlib import Path
import face_recognition
from dataset_filter.domain.interfaces import Detector
from dataset_filter.domain.models import FilterConfig
from dataset_filter.domain.models.detection_result import DetectionResult

class ScaleDetector(Detector):
    """Detects if the main face is large enough in the frame."""

    def __init__(self, config: FilterConfig) -> None:
        self._config = config

    def check(self, file_path: Path) -> DetectionResult:
        try:
            image = face_recognition.load_image_file(file_path)
            h, w = image.shape[:2]
            face_locations = face_recognition.face_locations(image)

            if not face_locations:
                return DetectionResult(is_defect=True, confidence=0.0, reason="No faces found for scale check")

            # Use the largest face area to determine scale
            max_area = 0
            for (top, right, bottom, left) in face_locations:
                area = (bottom - top) * (right - left)
                if area > max_area:
                    max_area = area

            img_area = h * w
            scale = max_area / img_area

            is_defect = scale < self._config.quality.scale_min
            # Confidence: 1.0 if above threshold, otherwise proportional to how close it is
            confidence = min(1.0, scale / self._config.quality.scale_min) if self._config.quality.scale_min > 0 else 1.0
            reason = "" if not is_defect else f"Face too small (scale: {scale:.4f})"

            return DetectionResult(is_defect=is_defect, confidence=confidence, reason=reason)
        except Exception as e:
            return DetectionResult(is_defect=True, confidence=0.0, reason=f"Scale detection error: {str(e)}")

    def get_weight(self) -> float:
        """Return the weight of this detector as defined in the configuration."""
        return self._config.quality.scale_weight
