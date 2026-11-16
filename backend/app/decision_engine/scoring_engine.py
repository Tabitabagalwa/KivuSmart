"""AHP-weighted fuzzy multi-criteria evaluation of restoration strategies.

For each strategy (RNA, reboisement pur, agroforesterie) every criterion yields a
membership score in [0, 1]; scores are combined with AHP weights. Missing inputs
are never replaced by defaults: they reduce ``coverage`` and confidence and are
listed as information gaps. The baseline (current practice) is diagnosed for
erosion risk and kept as the comparison reference.

``main()`` keeps the original command-line demo based on :mod:`backend.engine`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from KivuSmart.backend.engine import SCENARIO_DETAILS, build_sample_parcels, recommend_restoration_strategy

from . import rna
from . import agroforestry
from . import forestry
from .ahp import ahp_weights
from .criteria import CriterionResult, decreasing, increasing, number
from .rules import DEFAULT_RULES, load_species

ENGINE_VERSION = "2.0.0"
STRATEGY_MODULES = {module.STRATEGY_ID: module for module in (rna, forestry, agroforestry)}
LEGACY_REQUIRED = ("slope_pct", "ndvi", "distance_to_forest_m", "cropland")

VALIDATION_NOTE = (
    "Limites des données & validation locale : la résolution des capteurs (30 m pour "
    "HLS/NASADEM, 250 m MODIS, 9 km SMAP, ~50 km POWER) est souvent plus grossière que "
    "les parcelles. Toute recommandation doit être validée sur le terrain avec les "
    "agriculteurs et agronomes locaux avant plantation."
)

RESOLUTION_GAPS = {
    "soil_moisture": "SMAP (9 km) : une seule valeur couvre de nombreuses parcelles.",
    "root_zone_wetness": "NASA POWER (~50 km) : contexte régional uniquement.",
    "temperature_max_c": "NASA POWER (~50 km) : contexte régional uniquement.",
    "canopy_height_m": "GEDI échantillonne par empreintes discontinues de 25 m.",
}


def _weights_for(strategy_id: str, pairwise: list[list[float]] | None) -> tuple[list[float], dict]:
    module = STRATEGY_MODULES[strategy_id]
    matrix = pairwise if pairwise is not None else module.DEFAULT_PAIRWISE
    if len(matrix) != len(module.CRITERIA_ORDER):
        raise ValueError(
            f"La matrice AHP de {strategy_id} doit avoir {len(module.CRITERIA_ORDER)} critères "
            f"dans l'ordre {list(module.CRITERIA_ORDER)}."
        )
    result = ahp_weights(matrix)
    return result.weights, result.as_dict()


def _assess(strategy_id: str, metrics: dict, rules: dict, pairwise: list[list[float]] | None) -> dict:
    module = STRATEGY_MODULES[strategy_id]
    weights, ahp = _weights_for(strategy_id, pairwise)
    criteria: list[CriterionResult] = module.evaluate(metrics, rules)
    by_key = {c.key: c for c in criteria}
    weighted, available = 0.0, 0.0
    for key, weight in zip(module.CRITERIA_ORDER, weights):
        criterion = by_key[key]
        criterion.weight = round(weight, 4)
        if criterion.score is not None:
            weighted += weight * criterion.score
            available += weight
    min_coverage = rules["decision"]["min_coverage"]
    score = weighted / available if available > 0 else None
    status = "ok" if available >= min_coverage else "insufficient_data"
    details = SCENARIO_DETAILS[strategy_id]
    return {
        "strategy_id": strategy_id,
        "title": details["title"],
        "score": None if score is None else round(score, 4),
        "coverage": round(available, 4),
        "status": status,
        "ahp": ahp,
        "criteria": [c.as_dict() for c in criteria],
        "missing_variables": sorted({m for c in criteria for m in c.missing}),
    }


def baseline_diagnostic(metrics: dict, rules: dict) -> dict[str, Any]:
    r = rules["baseline"]
    factors = {
        "pente": increasing(number(metrics, "slope_pct"), r["slope_erosion_low_pct"], r["slope_erosion_high_pct"]),
        "sol_nu": decreasing(number(metrics, "ndvi"), r["ndvi_bare_max"], r["ndvi_bare_zero"]),
        "absence_couvert_arboré": decreasing(
            number(metrics, "canopy_height_m"), r["canopy_bare_max_m"], r["canopy_bare_zero_m"]
        ),
        "érosivité_pluies": increasing(number(metrics, "rainfall_mm"), r["rainfall_low_mm"], r["rainfall_high_mm"]),
    }
    available = [v for v in factors.values() if v is not None]
    risk = sum(available) / len(available) if available else None
    level = None if risk is None else ("élevé" if risk >= 0.66 else "modéré" if risk >= 0.33 else "faible")
    details = SCENARIO_DETAILS["baseline"]
    return {
        "strategy_id": "baseline",
        "title": details["title"],
        "erosion_risk_index": None if risk is None else round(risk, 4),
        "erosion_risk_level": level,
        "factors": {k: None if v is None else round(v, 4) for k, v in factors.items()},
        "summary": details["summary"],
        "risks": details["risks"],
    }


def _legacy(metrics: dict) -> dict[str, Any] | None:
    if any(number(metrics, key) is None for key in LEGACY_REQUIRED):
        return None
    clean = {k: number(metrics, k) for k in metrics if number(metrics, k) is not None}
    clean["cropland"] = bool(clean.get("cropland"))
    decision = recommend_restoration_strategy(clean)
    return {"strategy_id": decision["strategy_id"], "strategy": decision["strategy"]}


def evaluate_parcel(
    metrics: dict[str, Any],
    *,
    rules: dict | None = None,
    pairwise: dict[str, list[list[float]]] | None = None,
    reference_path: Path | None = None,
) -> dict[str, Any]:
    """Rank RNA / reboisement / agroforesterie for one parcel's metrics."""
    rules = rules or DEFAULT_RULES
    pairwise = pairwise or {}
    unknown = set(pairwise) - set(STRATEGY_MODULES)
    if unknown:
        raise ValueError(f"Stratégies inconnues dans les matrices AHP : {sorted(unknown)}")
    decision_rules = rules["decision"]

    assessments = [
        _assess(strategy_id, metrics, rules, pairwise.get(strategy_id))
        for strategy_id in STRATEGY_MODULES
    ]
    ranking = sorted(
        assessments,
        key=lambda a: (a["status"] == "ok", a["score"] if a["score"] is not None else -1),
        reverse=True,
    )
    eligible = [a for a in ranking if a["status"] == "ok" and a["score"] is not None]
    notes: list[str] = []

    if not eligible:
        status, chosen = "insufficient_data", None
        notes.append("Données insuffisantes pour comparer les stratégies : compléter les variables manquantes.")
    else:
        best = eligible[0]
        if best["score"] < decision_rules["min_score"]:
            status, chosen = "low_suitability", None
            notes.append(
                "Aucune stratégie n'atteint le score minimal : maintenir la pratique actuelle comme "
                "référence et organiser un diagnostic de terrain."
            )
        else:
            status, chosen = "recommended", best

    confidence = 0.0
    margin = None
    if chosen is not None:
        runner_up = next((a for a in eligible[1:]), None)
        margin = chosen["score"] - runner_up["score"] if runner_up else chosen["score"]
        confidence = chosen["coverage"] * (0.6 + 0.4 * min(1.0, margin / 0.15))
        if margin < decision_rules["tie_margin"]:
            notes.append(
                f"Scores proches entre {chosen['title']} et {runner_up['title']} : arbitrage à faire "
                "avec les exploitants."
            )
        if not chosen["ahp"]["consistent"]:
            confidence *= 0.8
            notes.append("Matrice AHP incohérente (RC > 0,10) : revoir les jugements de pondération.")

    used = {k for k in metrics if number(metrics, k) is not None}
    gaps = sorted({m for a in assessments for m in a["missing_variables"]})
    information_gaps = [f"Variable manquante : {g}" for g in gaps]
    information_gaps += [text for var, text in RESOLUTION_GAPS.items() if var in used]

    legacy = _legacy(metrics)
    if legacy is not None:
        legacy["agrees"] = chosen is not None and legacy["strategy_id"] == chosen["strategy_id"]

    recommended = None
    if chosen is not None:
        details = SCENARIO_DETAILS[chosen["strategy_id"]]
        recommended = {
            "strategy_id": chosen["strategy_id"],
            "title": details["title"],
            "summary": details["summary"],
            "benefits": details["benefits"],
            "risks": details["risks"],
            "score": chosen["score"],
            "margin_over_next": None if margin is None else round(margin, 4),
            "species": [
                s for s in load_species(reference_path)
                if chosen["strategy_id"] in s.get("recommended_for", [])
            ],
        }

    return {
        "engine_version": ENGINE_VERSION,
        "method": "Logique floue + pondération AHP (Saaty), agrégation somme pondérée",
        "decision_status": status,
        "recommended": recommended,
        "confidence": round(confidence, 4),
        "ranking": ranking,
        "baseline": baseline_diagnostic(metrics, rules),
        "legacy_rule_engine": legacy,
        "notes": notes,
        "information_gaps": information_gaps,
        "validation_note": VALIDATION_NOTE,
    }


def summarize_portfolio(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate parcel decisions; each result may carry ``area_ha``."""
    counts: dict[str, int] = {}
    areas: dict[str, float] = {}
    for item in results:
        evaluation = item["evaluation"]
        key = (
            evaluation["recommended"]["strategy_id"]
            if evaluation["recommended"] else evaluation["decision_status"]
        )
        counts[key] = counts.get(key, 0) + 1
        areas[key] = round(areas.get(key, 0.0) + float(item.get("area_ha") or 0.0), 4)
    return {"parcel_count": len(results), "by_strategy": counts, "area_ha_by_strategy": areas}


def main() -> None:
    print("Assessment of pilot parcels\n")
    for parcel in build_sample_parcels():
        decision = recommend_restoration_strategy(parcel)
        print(f"{parcel['name']}: {decision['strategy']}")
        print(f"  Summary: {decision['summary']}")
        evaluation = evaluate_parcel(parcel)
        top = evaluation["recommended"]
        label = top["title"] if top else evaluation["decision_status"]
        print(f"  AHP engine {ENGINE_VERSION}: {label} (confiance {evaluation['confidence']:.2f})")


if __name__ == "__main__":
    main()
