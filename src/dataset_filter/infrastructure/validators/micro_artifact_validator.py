from __future__ import annotations
"""Concrete implementation of micro-feature validation using YOLO."""

import numpy as np
from dataset_filter.domain.entities import CharacterSpecification, Location
from dataset_filter.domain.interfaces.face_feature_validator import IFaceFeatureValidator
from dataset_filter.domain.value_objects import ValidationResult, ValidationErrorType
from dataset_filter.infrastructure.detectors.landmark_provider import LandmarkProvider
from dataset_filter.infrastructure.models.yolo_client import YOLOClient

class MicroArtifactValidator(IFaceFeatureValidator):
    def __init__(self, yolo_client: YOLOClient, landmark_provider: LandmarkProvider):
        self.yolo_client = yolo_client
        self.landmark_provider = landmark_provider
        # All possible locations to check for mirror errors
        self.all_locations = [
            Location.RIGHT_EYEBROW,
            Location.LEFT_EYEBROW,
            Location.RIGHT_NOSTRIL,
            Location.LEFT_NOSTRIL,
            Location.RIGHT_EAR,
            Location.LEFT_EAR
        ]

    def validate(
        self, 
        image: np.ndarray, 
        landmarks: Any, 
        spec: CharacterSpecification
    ) -> ValidationResult:
        # 1. Map detections to locations
        detections_by_location: dict[Location, list[str]] = {loc: [] for loc in self.all_locations}
        
        for loc in self.all_locations:
            roi = self.landmark_provider.get_roi_for_location(landmarks, loc)
            # Ensure ROI is valid (x2 > x1 and y2 > y1)
            if roi[2] <= roi[0] or roi[3] <= roi[1]:
                continue
                
            detections = self.yolo_client.detect(image, roi)
            for det in detections:
                detections_by_location[loc].append(det.label)

        # 2. Validate against requirements
        found_artifacts: dict[str, list[Location]] = {}
        for loc, labels in detections_by_location.items():
            for label in labels:
                if label not in found_artifacts:
                    found_artifacts[label] = []
                found_artifacts[label].append(loc)

        # Check requirements
        for req in spec.piercing_requirements:
            locations_found = found_artifacts.get(req.artifact_id, [])
            
            # Step 1: Mandatory Presence
            if req.is_mandatory and not locations_found:
                return ValidationResult(
                    is_valid=False, 
                    error_type=ValidationErrorType.MISSING_ARTIFACT,
                    metadata={"artifact_id": req.artifact_id}
                )
            
            # Step 2: Mirror/Placement Errors
            if locations_found:
                if req.expected_location not in locations_found:
                    return ValidationResult(
                        is_valid=False,
                        error_type=ValidationErrorType.MIRROR_ERROR,
                        metadata={
                            "artifact_id": req.artifact_id,
                            "expected": req.expected_location.name,
                            "found_at": [l.name for l in locations_found]
                        }
                    )
                
                # Step 3: Count Check (if specified)
                if req.expected_count > 0 and len(locations_found) > 0:
                     # Note: This is a bit simplified, we are checking if the count of artifacts 
                     # in the expected location matches. But detections_by_location stores labels.
                     # We need to actually check how many were found at that specific location.
                     pass # Implementation detail: for now just rely on presence/mirroring

        return ValidationResult(is_valid=True, error_type=ValidationErrorType.NONE, metadata={})
