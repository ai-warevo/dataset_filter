from __future__ import annotations
"""Abstract interface for micro-feature validation."""

from abc import ABC, abstractmethod
from typing import Any
import numpy as np
from dataset_filter.domain.entities import CharacterSpecification
from dataset_filter.domain.value_objects import ValidationResult

# LandmarkSet can be a 68-point numpy array OR a dictionary of landmarks (e.g. from face_recognition)
LandmarkSet = Any

class IFaceFeatureValidator(ABC):
    @abstractmethod
    def validate(
        self, 
        image: np.ndarray, 
        landmarks: LandmarkSet, 
        spec: CharacterSpecification
    ) -> ValidationResult:
        """
        Performs feature-specific validation.
        Raises: Exception if model/detector fails (InfrastructureError should be used in real impl).
        """
        pass
