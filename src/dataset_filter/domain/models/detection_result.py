from __future__ import annotations
"""Domain value object representing the result of a detection check."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DetectionResult:
    """Result of a single detector check."""
    is_defect: bool
    confidence: float = 1.0
    reason: str = ""
    score: float | None = None
