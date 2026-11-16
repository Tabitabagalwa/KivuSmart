"""Trend analysis for parcel time series (NDVI, soil moisture, rainfall…).

Provides an ordinary least-squares slope (units per year) and the non-parametric
Mann-Kendall test with Sen's slope, which is robust to outliers and common in
vegetation-trend studies.
"""

from __future__ import annotations

import math
from datetime import date
from typing import Any

import numpy as np


def _normal_two_sided_p(z: float) -> float:
    return math.erfc(abs(z) / math.sqrt(2.0))


def mann_kendall(values: list[float]) -> dict[str, float]:
    n = len(values)
    s = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            s += int(np.sign(values[j] - values[i]))
    _, counts = np.unique(np.asarray(values), return_counts=True)
    tie_term = sum(int(t) * (int(t) - 1) * (2 * int(t) + 5) for t in counts if t > 1)
    variance = (n * (n - 1) * (2 * n + 5) - tie_term) / 18.0
    if variance <= 0:
        z = 0.0
    elif s > 0:
        z = (s - 1) / math.sqrt(variance)
    elif s < 0:
        z = (s + 1) / math.sqrt(variance)
    else:
        z = 0.0
    return {"s": float(s), "z": z, "p_value": _normal_two_sided_p(z)}


def sens_slope(times: np.ndarray, values: np.ndarray) -> float | None:
    slopes = [
        (values[j] - values[i]) / (times[j] - times[i])
        for i in range(len(values) - 1)
        for j in range(i + 1, len(values))
        if times[j] != times[i]
    ]
    return float(np.median(slopes)) if slopes else None


def analyze_series(
    points: list[tuple[date, float]],
    *,
    alpha: float = 0.05,
    min_points: int = 4,
) -> dict[str, Any]:
    """Summarize change between the first and last observation and test the trend."""
    clean = sorted((d, float(v)) for d, v in points if d is not None and math.isfinite(v))
    result: dict[str, Any] = {"n": len(clean), "alpha": alpha}
    if len(clean) < 2:
        result.update(status="insufficient_data",
                      message="Au moins deux observations datées sont nécessaires.")
        return result

    first_date, first_value = clean[0]
    last_date, last_value = clean[-1]
    change = last_value - first_value
    result.update(
        first={"date": first_date.isoformat(), "value": first_value},
        last={"date": last_date.isoformat(), "value": last_value},
        absolute_change=change,
        relative_change=(change / abs(first_value)) if first_value != 0 else None,
    )
    if len(clean) < min_points:
        result.update(status="change_only",
                      message=f"Tendance non testée : {min_points} observations minimum.")
        return result

    times = np.array([(d - first_date).days / 365.25 for d, _ in clean])
    values = np.array([v for _, v in clean])
    ols_slope = float(np.polyfit(times, values, 1)[0]) if np.ptp(times) > 0 else None
    mk = mann_kendall(values.tolist())
    significant = mk["p_value"] < alpha
    if not significant:
        direction = "aucune tendance significative"
    else:
        direction = "hausse" if mk["s"] > 0 else "baisse"
    result.update(
        status="tested",
        ols_slope_per_year=ols_slope,
        sens_slope_per_year=sens_slope(times, values),
        mann_kendall=mk,
        significant=significant,
        trend=direction,
    )
    return result


def climate_anomaly(
    records: list[dict[str, Any]],
    baseline_years: tuple[int, int],
    target_years: tuple[int, int],
) -> dict[str, Any]:
    """Compare a target period to a baseline for POWER monthly records (period=YYYYMM)."""
    def window(start: int, end: int) -> list[float]:
        return [
            r["value"] for r in records
            if r.get("value") is not None and start <= int(str(r["period"])[:4]) <= end
        ]

    baseline = window(*baseline_years)
    target = window(*target_years)
    if not baseline or not target:
        return {"status": "insufficient_data"}
    baseline_mean = float(np.mean(baseline))
    target_mean = float(np.mean(target))
    baseline_std = float(np.std(baseline, ddof=1)) if len(baseline) > 1 else None
    return {
        "status": "ok",
        "baseline_mean": baseline_mean,
        "target_mean": target_mean,
        "anomaly": target_mean - baseline_mean,
        "relative_anomaly": (target_mean - baseline_mean) / abs(baseline_mean) if baseline_mean else None,
        "standardized_anomaly": (target_mean - baseline_mean) / baseline_std if baseline_std else None,
    }
