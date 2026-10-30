from __future__ import annotations
"""Value Object encapsulating threshold parameters and strict type policies."""

from dataclasses import dataclass, field
from enum import Enum, auto


class EvaluationMode(Enum):
    STRICT = auto()
    SCORING = auto()


@dataclass(frozen=True)
class MatchConfig:
    tolerance: float = 0.55
    weight: float = 1.0
    deduplication_threshold: float = 0.4  # Euclidean distance threshold for duplicates


@dataclass(frozen=True)
class QualityConfig:
    blur_threshold: float = 120.0
    blur_weight: float = 0.3
    face_count_max: int = 1
    face_count_weight: float = 1.0
    scale_min: float = 0.1
    scale_weight: float = 0.5


@dataclass(frozen=True)
class FilterConfig:
    evaluation_mode: EvaluationMode = EvaluationMode.STRICT
    global_threshold: float = 0.65
    valid_extensions: tuple[str, ...] = (".png", ".jpg", ".jpeg")
    match: MatchConfig = field(default_factory=MatchConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
