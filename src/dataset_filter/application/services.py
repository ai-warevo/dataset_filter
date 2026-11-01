from __future__ import annotations
"""Service for orchestrating validation processes."""

from typing import List
import numpy as np
from dataset_filter.domain.entities import CharacterSpecification
from dataset_filter.domain.interfaces.face_feature_validator import IFaceFeatureValidator
from dataset_filter.domain.value_objects import ValidationResult

class ValidationOrchestrator:
    def __init__(self, validators: List[IFaceFeatureValidator]):
        self.validators = validators

    def run_all(
        self, 
        image: np.ndarray, 
        landmarks: np.ndarray, 
        spec: CharacterSpecification
    ) -> list[ValidationResult]:
        """Runs all registered validators and returns their results."""
        results = []
        for validator in self.validators:
            try:
                result = validator.validate(image, landmarks, spec)
                results.append(result)
            except Exception as e:
                # In a real system, we'd log this and perhaps wrap it in an InfrastructureError
                print(f"Validator {validator.__class__.__name__} failed: {e}")
        return results
