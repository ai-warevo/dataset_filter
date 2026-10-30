from __future__ import annotations
"""Value Object encapsulating threshold parameters and strict type policies."""

from dataclasses import dataclass


@dataclass(frozen=True)
class FilterConfig:
    tolerance: float = 0.55
    valid_extensions: tuple[str, ...] = (".png", ".jpg", ".jpeg")
