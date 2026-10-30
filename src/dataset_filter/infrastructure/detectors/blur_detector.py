from __future__ import annotations
from pathlib import Path
import cv2
import face_recognition
from dataset_filter.domain.interfaces import Detector
from dataset_filter.domain.models import FilterConfig
from dataset_filter.domain.models.detection_result import DetectionResult

class BlurDetector(Detector):
    """Detects if the image is too blurry using Laplacian variance."""

    def __init__(self, config: FilterConfig) -> None:
        self._config = config

    def check(self, file_path: Path) -> DetectionResult:
        try:
            # Use face_recognition to load to avoid OpenCV path issues on Windows
            image = face_recognition.load_image_file(file_path)
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            variance = cv2.Laplacian(gray, cv2.CV_64F).var()

            is_defect = variance < self._config.quality.blur_threshold
            # Confidence: 1.0 if above threshold, otherwise proportional to how close it is
            confidence = min(1.0, variance / self._config.quality.blur_threshold) if self._config.quality.blur_threshold > 0 else 1.0
            reason = "" if not is_defect else f"Image too blurry (variance: {variance:.2f})"

            return DetectionResult(is_defect=is_defect, confidence=confidence, reason=reason)
        except Exception as e:
            return DetectionResult(is_defect=True, confidence=0.0, reason=f"Blur detection error: {str(e)}")
