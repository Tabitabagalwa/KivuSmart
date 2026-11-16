"""Régénération naturelle assistée (RNA) suitability criteria.

Brief: slope 15–35 %, < 800 m from a forest or corridor, biological potential
(NDVI 0.30–0.50 or GEDI canopy height > 1.5 m).
"""

from __future__ import annotations

from typing import Any

from .criteria import CriterionResult, best_of, decreasing, increasing, number, trapezoid

STRATEGY_ID = "rna"
CRITERIA_ORDER = ("slope", "forest_proximity", "biological_potential", "land_use")
# Saaty pairwise judgements, rows/columns in CRITERIA_ORDER.
DEFAULT_PAIRWISE = [
    [1, 1 / 2, 1 / 2, 2],
    [2, 1, 1, 3],
    [2, 1, 1, 3],
    [1 / 2, 1 / 3, 1 / 3, 1],
]


def evaluate(metrics: dict[str, Any], rules: dict[str, Any]) -> list[CriterionResult]:
    r = rules["rna"]
    slope = number(metrics, "slope_pct")
    forest = number(metrics, "distance_to_forest_m")
    ndvi = number(metrics, "ndvi")
    canopy = number(metrics, "canopy_height_m")
    cropland = number(metrics, "cropland")

    ndvi_score = trapezoid(
        ndvi, r["ndvi_min"] - r["ndvi_margin"], r["ndvi_min"], r["ndvi_max"], r["ndvi_max"] + r["ndvi_margin"]
    )
    canopy_score = increasing(canopy, r["canopy_height_zero_m"], r["canopy_height_min_m"])
    return [
        CriterionResult(
            key="slope",
            label="Pente modérée",
            score=trapezoid(
                slope,
                r["slope_min_pct"] - r["slope_margin_pct"],
                r["slope_min_pct"],
                r["slope_max_pct"],
                r["slope_max_pct"] + r["slope_margin_pct"],
            ),
            inputs={"slope_pct": slope},
            rationale=f"Optimale entre {r['slope_min_pct']:g} et {r['slope_max_pct']:g} %.",
            evidence="MNT NASADEM/SRTM 30 m",
            missing=[] if slope is not None else ["slope_pct"],
        ),
        CriterionResult(
            key="forest_proximity",
            label="Proximité d'une forêt ou d'un corridor",
            score=decreasing(forest, r["forest_distance_max_m"], r["forest_distance_zero_m"]),
            inputs={"distance_to_forest_m": forest},
            rationale=(
                f"Dispersion naturelle des graines : idéal < {r['forest_distance_max_m']:g} m, "
                f"nul au-delà de {r['forest_distance_zero_m']:g} m."
            ),
            evidence="Couche forêts/corridors (PNKB, RNI, bosquets) + distance UTM 35S",
            missing=[] if forest is not None else ["distance_to_forest_m"],
        ),
        CriterionResult(
            key="biological_potential",
            label="Potentiel biologique (souches, recrû)",
            score=best_of(ndvi_score, canopy_score),
            inputs={"ndvi": ndvi, "canopy_height_m": canopy},
            rationale=(
                f"NDVI moyen {r['ndvi_min']:g}–{r['ndvi_max']:g} ou canopée GEDI > "
                f"{r['canopy_height_min_m']:g} m."
            ),
            evidence="NDVI HLS L30 / MODIS ; hauteur de canopée GEDI L2A",
            missing=[k for k, v in (("ndvi", ndvi), ("canopy_height_m", canopy)) if v is None]
            if ndvi_score is None or canopy_score is None else [],
        ),
        CriterionResult(
            key="land_use",
            label="Absence de culture active",
            score=None if cropland is None else (r["cropland_score"] if cropland >= 0.5 else 1.0),
            inputs={"cropland": cropland},
            rationale="La RNA suppose de laisser la végétation se régénérer sans labour.",
            evidence="Occupation du sol (MCD12Q1) et enquête locale",
            missing=[] if cropland is not None else ["cropland"],
        ),
    ]
