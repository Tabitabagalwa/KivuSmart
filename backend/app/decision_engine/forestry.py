"""Reboisement pur (critical protection) suitability criteria.

Brief: critical slope > 35 %, < 300 m from springs/headwaters, strongly degraded
soil (NDVI < 0.25).
"""

from __future__ import annotations

from typing import Any

from .criteria import CriterionResult, best_of, decreasing, increasing, number

STRATEGY_ID = "reboisement"
CRITERIA_ORDER = ("critical_slope", "water_source_buffer", "degradation", "non_agricultural")
DEFAULT_PAIRWISE = [
    [1, 2, 2, 4],
    [1 / 2, 1, 1, 3],
    [1 / 2, 1, 1, 3],
    [1 / 4, 1 / 3, 1 / 3, 1],
]


def evaluate(metrics: dict[str, Any], rules: dict[str, Any]) -> list[CriterionResult]:
    r = rules["reboisement"]
    slope = number(metrics, "slope_pct")
    water = number(metrics, "water_buffer_m")
    ndvi = number(metrics, "ndvi")
    degraded = number(metrics, "degraded")
    cropland = number(metrics, "cropland")

    ndvi_score = decreasing(ndvi, r["ndvi_degraded_max"], r["ndvi_degraded_zero"])
    flag_score = None if degraded is None else (1.0 if degraded >= 0.5 else 0.0)
    return [
        CriterionResult(
            key="critical_slope",
            label="Pente critique",
            score=increasing(slope, r["slope_min_pct"] - r["slope_ramp_pct"], r["slope_min_pct"]),
            inputs={"slope_pct": slope},
            rationale=f"Risque d'érosion et de glissement au-delà de {r['slope_min_pct']:g} %.",
            evidence="MNT NASADEM/SRTM 30 m",
            missing=[] if slope is not None else ["slope_pct"],
        ),
        CriterionResult(
            key="water_source_buffer",
            label="Zone tampon des sources",
            score=decreasing(water, r["water_buffer_m"], r["water_buffer_zero_m"]),
            inputs={"water_buffer_m": water},
            rationale=f"Protection prioritaire à moins de {r['water_buffer_m']:g} m d'une source.",
            evidence="Réseau hydrographique / sources (couche locale) + distance UTM 35S",
            missing=[] if water is not None else ["water_buffer_m"],
        ),
        CriterionResult(
            key="degradation",
            label="Dégradation du sol et du couvert",
            score=best_of(ndvi_score, flag_score),
            inputs={"ndvi": ndvi, "degraded": degraded},
            rationale=f"Sol très dégradé si NDVI < {r['ndvi_degraded_max']:g}.",
            evidence="NDVI HLS L30 / MODIS ; diagnostic de terrain",
            missing=[k for k, v in (("ndvi", ndvi), ("degraded", degraded)) if v is None]
            if ndvi_score is None and flag_score is None else [],
        ),
        CriterionResult(
            key="non_agricultural",
            label="Hors usage agricole",
            score=None if cropland is None else (r["cropland_score"] if cropland >= 0.5 else 1.0),
            inputs={"cropland": cropland},
            rationale="Le reboisement pur retire la parcelle des cultures annuelles.",
            evidence="Occupation du sol (MCD12Q1) et enquête locale",
            missing=[] if cropland is not None else ["cropland"],
        ),
    ]
