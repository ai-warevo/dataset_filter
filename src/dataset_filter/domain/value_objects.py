from __future__ import annotations
"""Value Objects for validation outcomes."""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict

class ValidationErrorType(Enum):
    NONE = "none"
    MISSING_ARTIFACT = "missing_artifact"
    MIRROR_ERROR = "mirror_error"
    PLACEMENT_ERROR = "placement_error"
    MISMATCHED_COUNT = "mismatched_count"

@dataclass(frozen=True)
class ValidationResult:
    is_valid: bool
    error_type: ValidationErrorType
    metadata: Dict[str, Any]  # e.g., {"detected_at": "LEFT_EYEBROW", "expected": "RIGHT_EYEBROW"}
