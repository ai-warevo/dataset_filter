from __future__ import annotations
"""Domain entities representing character specifications."""

import json
from dataclasses import dataclass
from enum import Enum, auto
from typing import List

class Location(Enum):
    RIGHT_EYEBROW = auto()
    LEFT_EYEBROW = auto()
    RIGHT_NOSTRIL = auto()
    LEFT_NOSTRIL = auto()
    RIGHT_EAR = auto()
    LEFT_EAR = auto()

@dataclass(frozen=True)
class PiercingRequirement:
    artifact_id: str              # e.g., "silver_hoop"
    expected_location: Location  # Enum: RIGHT_EYEBROW, LEFT_NOSTRIL, etc.
    is_mandatory: bool            # If True, absence triggers Fail
    expected_count: int = 0       # Number of artifacts expected in this location

@dataclass(frozen=True)
class CharacterSpecification:
    character_id: str
    piercing_requirements: List[PiercingRequirement]

    @classmethod
    def from_json(cls, json_str: str) -> CharacterSpecification:
        data = json.loads(json_str)
        reqs = []
        for r in data["piercing_requirements"]:
            reqs.append(PiercingRequirement(
                artifact_id=r["artifact_id"],
                expected_location=Location[r["expected_location"]],
                is_mandatory=r["is_mandatory"],
                expected_count=r.get("expected_count", 0)
            ))
        return cls(
            character_id=data["character_id"],
            piercing_requirements=reqs
        )
