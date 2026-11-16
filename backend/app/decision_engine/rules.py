"""Decision thresholds.

Priority (lowest to highest): built-in defaults from the challenge brief, the
``thresholds`` of ``data/reference/referentiel_kivu.json``, then
``config/restoration_rules.json`` (same nested structure as ``DEFAULT_RULES``).
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any

DEFAULT_RULES: dict[str, dict[str, float]] = {
    "rna": {
        "slope_min_pct": 15, "slope_max_pct": 35, "slope_margin_pct": 5,
        "forest_distance_max_m": 800, "forest_distance_zero_m": 2000,
        "ndvi_min": 0.30, "ndvi_max": 0.50, "ndvi_margin": 0.05,
        "canopy_height_min_m": 1.5, "canopy_height_zero_m": 1.0,
        "cropland_score": 0.2,
    },
    "reboisement": {
        "slope_min_pct": 35, "slope_ramp_pct": 10,
        "water_buffer_m": 300, "water_buffer_zero_m": 600,
        "ndvi_degraded_max": 0.25, "ndvi_degraded_zero": 0.40,
        "cropland_score": 0.2,
    },
    "agroforesterie": {
        "slope_max_pct": 20, "slope_zero_pct": 30,
        "soil_moisture_deficit_m3m3": 0.25, "soil_moisture_ok_m3m3": 0.35,
        "root_wetness_deficit": 0.5, "root_wetness_ok": 0.7,
        "tmax_stress_low_c": 27, "tmax_stress_high_c": 32,
        "lst_stress_low_c": 30, "lst_stress_high_c": 38,
        "village_distance_max_km": 2.0, "village_distance_zero_km": 5.0,
    },
    "baseline": {
        "slope_erosion_low_pct": 15, "slope_erosion_high_pct": 35,
        "ndvi_bare_max": 0.25, "ndvi_bare_zero": 0.40,
        "canopy_bare_max_m": 1.0, "canopy_bare_zero_m": 2.0,
        "rainfall_low_mm": 1200, "rainfall_high_mm": 1800,
    },
    "decision": {"min_score": 0.5, "min_coverage": 0.6, "tie_margin": 0.05},
}

# referentiel_kivu.json "thresholds" keys -> (section, key)
REFERENTIEL_MAPPING = {
    "rna_slope_min_pct": ("rna", "slope_min_pct"),
    "rna_slope_max_pct": ("rna", "slope_max_pct"),
    "reboisement_slope_min_pct": ("reboisement", "slope_min_pct"),
    "agroforesterie_slope_max_pct": ("agroforesterie", "slope_max_pct"),
    "forest_distance_max_m": ("rna", "forest_distance_max_m"),
    "village_distance_max_km": ("agroforesterie", "village_distance_max_km"),
}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Fichier de règles illisible {path}: {exc}") from exc
    return data if isinstance(data, dict) else {}


def _set(rules: dict, section: str, key: str, value: Any, origin: str) -> None:
    if section not in rules or key not in rules[section]:
        raise ValueError(f"Règle inconnue dans {origin} : {section}.{key}")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"Valeur numérique attendue pour {section}.{key} ({origin}).")
    rules[section][key] = float(value)


def merge_rules(base: dict, overrides: dict[str, Any], origin: str = "overrides") -> dict:
    merged = copy.deepcopy(base)
    for section, values in (overrides or {}).items():
        if not isinstance(values, dict):
            raise ValueError(f"Section de règles invalide dans {origin} : {section}")
        for key, value in values.items():
            _set(merged, section, key, value, origin)
    return merged


def load_rules(config_dir: Path | None = None, reference_path: Path | None = None) -> dict:
    rules = copy.deepcopy(DEFAULT_RULES)
    if reference_path is not None:
        thresholds = _read_json(reference_path).get("thresholds", {})
        for name, value in thresholds.items():
            if name in REFERENTIEL_MAPPING:
                _set(rules, *REFERENTIEL_MAPPING[name], value, str(reference_path))
    if config_dir is not None:
        path = config_dir / "restoration_rules.json"
        rules = merge_rules(rules, _read_json(path), str(path))
    return rules


def load_species(reference_path: Path | None) -> list[dict[str, Any]]:
    if reference_path is None:
        return []
    species = _read_json(reference_path).get("species", [])
    return species if isinstance(species, list) else []
