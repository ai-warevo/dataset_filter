from __future__ import annotations
"""Heuristic implementation of micro-feature validation using OpenCV morphology."""

import cv2
import numpy as np
from dataset_filter.domain.entities import CharacterSpecification, Location
from dataset_filter.domain.interfaces.face_feature_validator import IFaceFeatureValidator
from dataset_filter.domain.value_objects import ValidationResult, ValidationErrorType
from dataset_filter.infrastructure.detectors.landmark_provider import LandmarkProvider
from dataset_filter.domain.models.filter_config import HeuristicConfig
from typing import Any

class HeuristicFaceFeatureValidator(IFaceFeatureValidator):
    def __init__(self, landmark_provider: LandmarkProvider, params: HeuristicConfig):
        self._landmark_provider = landmark_provider
        self.params = params

    def validate(
        self, 
        image: np.ndarray, 
        landmarks: Any, 
        spec: CharacterSpecification
    ) -> ValidationResult:
        """Validates micro-features (piercings) using morphological detection."""
        findings = {}
        errors = []

        for req in spec.piercing_requirements:
            loc = req.expected_location
            roi = self._landmark_provider.get_roi_for_location(landmarks, loc, image)
            count = 0

            if roi[2] > roi[0] and roi[3] > roi[1]:
                count = self._count_artifacts(image, roi, loc)

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

    def _count_artifacts(self, image: np.ndarray, roi: tuple[int, int, int, int], location: Location) -> int:
        """Counts artifacts (bright piercings or dark eyebrows) using Morphological transforms."""
        crop = image[roi[1]:roi[3], roi[0]:roi[2]]
        if crop.size == 0:
            return 0

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        
        # Determine if we are looking for dark features (eyebrows) or bright ones (piercings)
        is_dark_feature = any(word in location.name.upper() for word in ["EYEBROW", "EYE", "LASH"])
        
        # Use a slightly larger kernel for closing to ensure fragments merge (fixes over-counting)
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.params.kernel_size + 2, self.params.kernel_size + 2))
        # Use the standard kernel for the initial transform
        transform_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.params.kernel_size, self.params.kernel_size))

        if is_dark_feature:
            # Black-Hat extracts dark features from a lighter background
            transformed = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, transform_kernel)
        else:
            # Top-Hat extracts bright features from a darker background
            transformed = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, transform_kernel)
        
        # 1. Threshold the transformed image to find candidate regions
        _, thresh = cv2.threshold(transformed, 10, 255, cv2.THRESH_BINARY)
        
        # 2. Close gaps to prevent one artifact from being multiple contours (fixes over-counting)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, close_kernel)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        valid_count = 0
        for cnt in contours:
            if cv2.contourArea(cnt) > self.params.min_contour_area:
                # 3. Final validation
                mask = np.zeros(gray.shape, dtype=np.uint8)
                cv2.drawContours(mask, [cnt], -1, 255, -1)
                mean_val = np.mean(gray[mask == 255])

                if is_dark_feature:
                    # For dark features, we just need to ensure they were detected via contrast
                    valid_count += 1
                else:
                    # For bright features, ensure absolute brightness meets the floor
                    if mean_val > self.params.min_brightness_floor:
                        valid_count += 1

        return valid_count
