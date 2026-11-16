"""Fuzzy membership functions and criterion result container."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class CriterionResult:
    key: str
    label: str
    score: float | None
    inputs: dict[str, Any]
    rationale: str
    evidence: str
    weight: float = 0.0
    missing: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["score"] = None if self.score is None else round(self.score, 4)
        return data


def number(metrics: dict[str, Any], key: str) -> float | None:
    value = metrics.get(key)
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, str):
        lowered = value.strip().casefold()
        if lowered in {"true", "oui", "yes", "vrai"}:
            return 1.0
        if lowered in {"false", "non", "no", "faux"}:
            return 0.0
        value = lowered.replace(",", ".")
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def increasing(x: float | None, low: float, high: float) -> float | None:
    """0 at or below ``low``, 1 at or above ``high``, linear in between."""
    if x is None:
        return None
    if x <= low:
        return 0.0
    if x >= high:
        return 1.0
    return (x - low) / (high - low)


def decreasing(x: float | None, low: float, high: float) -> float | None:
    """1 at or below ``low``, 0 at or above ``high``, linear in between."""
    value = increasing(x, low, high)
    return None if value is None else 1.0 - value


def trapezoid(x: float | None, a: float, b: float, c: float, d: float) -> float | None:
    """0 outside [a, d], 1 within [b, c], linear on the shoulders."""
    if x is None:
        return None
    if x < a or x > d:
        return 0.0
    if b <= x <= c:
        return 1.0
    if x < b:
        return (x - a) / (b - a) if b > a else 1.0
    return (d - x) / (d - c) if d > c else 1.0


def best_of(*scores: float | None) -> float | None:
    available = [s for s in scores if s is not None]
    return max(available) if available else None
