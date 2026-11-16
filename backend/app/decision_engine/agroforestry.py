"""Agroforesterie intelligente suitability criteria.

Brief: moderate slope < 20 %, active cropland with moisture deficit (SMAP) or
heat stress (ECOSTRESS), < 2 km from villages and roads.
"""

from __future__ import annotations

from typing import Any

from .criteria import CriterionResult, best_of, decreasing, increasing, number

STRATEGY_ID = "agroforesterie"
CRITERIA_ORDER = ("moderate_slope", "active_cropland", "water_or_heat_stress", "accessibility")
DEFAULT_PAIRWISE = [
    [1, 1 / 2, 1, 2],
    [2, 1, 2, 3],
    [1, 1 / 2, 1, 2],
    [1 / 2, 1 / 3, 1 / 2, 1],
]


def evaluate(metrics: dict[str, Any], rules: dict[str, Any]) -> list[CriterionResult]:
    r = rules["agroforesterie"]
    slope = number(metrics, "slope_pct")
    cropland = number(metrics, "cropland")
    moisture = number(metrics, "soil_moisture")
    wetness = number(metrics, "root_zone_wetness")
    tmax = number(metrics, "temperature_max_c")
    lst = number(metrics, "land_surface_temperature_c")
    village = number(metrics, "distance_to_village_km")

    stress = best_of(
        decreasing(moisture, r["soil_moisture_deficit_m3m3"], r["soil_moisture_ok_m3m3"]),
        decreasing(wetness, r["root_wetness_deficit"], r["root_wetness_ok"]) if moisture is None else None,
        increasing(lst, r["lst_stress_low_c"], r["lst_stress_high_c"]),
        increasing(tmax, r["tmax_stress_low_c"], r["tmax_stress_high_c"]),
    )
    stress_inputs = {
        "soil_moisture": moisture,
        "root_zone_wetness": wetness,
        "land_surface_temperature_c": lst,
        "temperature_max_c": tmax,
    }
    return [
        CriterionResult(
            key="moderate_slope",
            label="Pente modérée",
            score=decreasing(slope, r["slope_max_pct"], r["slope_zero_pct"]),
            inputs={"slope_pct": slope},
            rationale=f"Cultures associées possibles sous {r['slope_max_pct']:g} %.",
            evidence="MNT NASADEM/SRTM 30 m",
            missing=[] if slope is not None else ["slope_pct"],
        ),
        CriterionResult(
            key="active_cropland",
            label="Terre agricole active",
            score=None if cropland is None else (1.0 if cropland >= 0.5 else 0.0),
            inputs={"cropland": cropland},
            rationale="L'agroforesterie concilie revenu paysan et régénération des sols.",
            evidence="Occupation du sol (MCD12Q1) et enquête locale",
            missing=[] if cropland is not None else ["cropland"],
        ),
        CriterionResult(
            key="water_or_heat_stress",
            label="Déficit hydrique ou stress thermique",
            score=stress,
            inputs=stress_inputs,
            rationale=(
                f"Humidité SMAP < {r['soil_moisture_deficit_m3m3']:g} m³/m³ ou stress thermique "
                "des cultures non ombragées."
            ),
            evidence="SMAP L3 (9 km), ECOSTRESS LST (70 m), NASA POWER (~50 km)",
            missing=[] if stress is not None else list(stress_inputs),
        ),
        CriterionResult(
            key="accessibility",
            label="Accessibilité (villages, routes)",
            score=decreasing(village, r["village_distance_max_km"], r["village_distance_zero_km"]),
            inputs={"distance_to_village_km": village},
            rationale=f"Suivi et entretien facilités à moins de {r['village_distance_max_km']:g} km.",
            evidence="Localités et routes (couche locale) + distance UTM 35S",
            missing=[] if village is not None else ["distance_to_village_km"],
        ),
    ]
