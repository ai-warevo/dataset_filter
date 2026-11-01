from __future__ import annotations
"""Heuristic implementation of micro-feature validation using OpenCV contrast and brightness."""

import cv2
import numpy as np
from dataset_filter.domain.entities import CharacterSpecification, Location
from dataset_filter.domain.interfaces.face_feature_validator import IFaceFeatureValidator
from dataset_filter.domain.value_objects import ValidationResult, ValidationErrorType
from dataset_filter.infrastructure.detectors.landmark_provider import LandmarkProvider
from dataset_filter.domain.models.filter_config import HeuristicConfig
from typing import Any, Dict

class HeuristicMetalValidator(IFaceFeatureValidator):
    def __init__(self, landmark_provider: LandmarkProvider, params: HeuristicConfig):
        self._landmark_provider = landmark_provider
        self.params = params

    def validate(
        self, 
        image: np.ndarray, 
        landmarks: Any, 
        spec: CharacterSpecification
    ) -> ValidationResult:
        """Validates piercings using morphological top-hat detection."""
        findings = {}
        errors = []

        for req in spec.piercing_requirements:
            loc = req.expected_location
            roi = self._landmark_provider.get_roi_for_location(landmarks, loc, image)
            count = 0

            if roi[2] > roi[0] and roi[3] > roi[1]:
                count = self._count_artifacts(image, roi)

            findings[loc.name] = {
                "count": count,
                "bbox": roi[:4]
            }

            if count != req.expected_count and req.is_mandatory:
                errors.append({
                    "location": loc.name,
                    "expected": req.expected_count,
                    "found": count
                })

        if errors:
            first_err = errors[0]
            return ValidationResult(
                is_valid=False,
                error_type=ValidationErrorType.MISSING_ARTIFACT,
                metadata={
                    "findings": findings, 
                    "error": {
                        "location": first_err["location"],
                        "expected": first_err["expected"],
                        "found": first_err["found"]
                    }
                }
            )

        return ValidationResult(is_valid=True, error_type=ValidationErrorType.NONE, metadata={"findings": findings})

    def _count_artifacts(self, image: np.ndarray, roi: tuple[int, int, int, int]) -> int:
        """Counts bright artifacts using Morphological Top-Hat transform."""
        crop = image[roi[1]:roi[3], roi[0]:roi[2]]
        if crop.size == 0:
            return 0

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        
        # 1. Morphological Top-Hat Transform
        # This extracts bright features that are smaller than the kernel size.
        # It effectively removes the background (skin/texture) and leaves only the highlights.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.params.kernel_size, self.params.kernel_size))
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)

        # 2. Thresholding the Top-Hat result
        # Since background is now ~0, we can use a much more stable threshold.
        # We use the multiplier on the top-hat's own intensity to find significant peaks.
        _, thresh = cv2.threshold(tophat, self.params.min_brightness_floor, 255, cv2.THRESH_BINARY)

        # 3. Contour Analysis
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        valid_count = 0
        for cnt in contours:
            if cv2.contourArea(cnt) > self.params.min_contour_area:
                # Final check: is the actual pixel intensity high enough?
                # (Top-hat only shows contrast, we still want to ensure it's actually bright)
                mask = np.zeros(gray.shape, dtype=np.uint8)
                cv2.drawContours(mask, [cnt], -1, 255, -1)
                if np.mean(gray[mask == 255]) > self.params.min_brightness_floor:
                    valid_count += 1

        return valid_count
