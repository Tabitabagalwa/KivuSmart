from __future__ import annotations

from typing import Any, Dict, Iterable, List


SCENARIO_DETAILS = {
    "baseline": {
        "title": "Pratique actuelle (Baseline)",
        "summary": "Culture annuelle continue dans des zones exposées au ruissellement et à la dégradation du sol.",
        "benefits": ["Référence de comparaison", "Diagnostic immédiat du paysage actuel"],
        "risks": ["Érosion accélérée", "Perte de matière organique", "Stress hydrique"],
    },
    "rna": {
        "title": "Régénération naturelle assistée (RNA)",
        "summary": "Idéal dans les zones de recrû avec pente modérée et proximité d'une forêt ou d'un corridor de végétation.",
        "benefits": ["Coût faible", "Restauration rapide de la biodiversité", "Protection des sols sans labour"],
        "risks": ["Nécessite une validation de terrain", "Dépend de la présence de souches vivantes"],
    },
    "reboisement": {
        "title": "Reboisement pur",
        "summary": "Recommandé sur versants très pentus et hors usage agricole pour stabiliser les sols et réduire l'érosion.",
        "benefits": ["Stabilisation mécanique des versants", "Recharge des aquifères", "Réduction du ravinement"],
        "risks": ["Coût d'investissement plus élevé", "Protection des sols avant toute culture"],
    },
    "agroforesterie": {
        "title": "Agroforesterie intelligente",
        "summary": "Mieux adaptée aux terres agricoles avec pente modérée, déficit d'humidité et accès aux villages.",
        "benefits": ["Revenus agricoles immédiats", "Sécurité alimentaire", "Amélioration de l'humidité et de la qualité du sol"],
        "risks": ["Exige des choix d'espèces adaptés au contexte", "Nécessite un accompagnement agronomique"],
    },
}


def build_sample_parcels() -> List[Dict[str, Any]]:
    return [
        {
            "id": "parcelle_01",
            "name": "Parcelle pilote Nord",
            "slope_pct": 18,
            "ndvi": 0.46,
            "distance_to_forest_m": 550,
            "distance_to_village_km": 1.2,
            "distance_to_field_m": 180,
            "distance_to_protected_area_km": 0.9,
            "soil_moisture": 0.42,
            "rainfall_mm": 1500,
            "temperature_c": 24.1,
            "canopy_height_m": 2.4,
            "water_buffer_m": 320,
            "cropland": True,
            "degraded": False,
        },
        {
            "id": "parcelle_02",
            "name": "Versant critique Sud",
            "slope_pct": 42,
            "ndvi": 0.20,
            "distance_to_forest_m": 2000,
            "distance_to_village_km": 4.1,
            "distance_to_field_m": 910,
            "distance_to_protected_area_km": 8.0,
            "soil_moisture": 0.21,
            "rainfall_mm": 1800,
            "temperature_c": 22.8,
            "canopy_height_m": 1.2,
            "water_buffer_m": 140,
            "cropland": False,
            "degraded": True,
        },
        {
            "id": "parcelle_03",
            "name": "Terroir agricole Ouest",
            "slope_pct": 12,
            "ndvi": 0.33,
            "distance_to_forest_m": 1500,
            "distance_to_village_km": 1.8,
            "distance_to_field_m": 90,
            "distance_to_protected_area_km": 3.5,
            "soil_moisture": 0.26,
            "rainfall_mm": 1300,
            "temperature_c": 25.3,
            "canopy_height_m": 1.7,
            "water_buffer_m": 430,
            "cropland": True,
            "degraded": True,
        },
    ]


def score_parcel(parcel: Dict[str, Any]) -> Dict[str, Any]:
    slope_pct = float(parcel.get("slope_pct", 0))
    ndvi = float(parcel.get("ndvi", 0.0))
    soil_moisture = float(parcel.get("soil_moisture", 0.0))
    forest_distance_m = float(parcel.get("distance_to_forest_m", 9999))
    village_distance_km = float(parcel.get("distance_to_village_km", 9999))
    field_distance_m = float(parcel.get("distance_to_field_m", 9999))
    protected_area_distance_km = float(parcel.get("distance_to_protected_area_km", 9999))
    water_buffer_m = float(parcel.get("water_buffer_m", 9999))

    slope_score = 0.0
    if 15 <= slope_pct <= 35:
        slope_score = 5.0
    elif slope_pct < 15:
        slope_score = 3.0
    elif slope_pct > 35:
        slope_score = 1.0

    forest_score = 5.0 if forest_distance_m <= 800 else 2.0 if forest_distance_m <= 1500 else 1.0
    village_score = 5.0 if village_distance_km <= 2.0 else 3.0 if village_distance_km <= 5 else 1.0
    field_score = 5.0 if field_distance_m <= 200 else 3.0 if field_distance_m <= 700 else 1.0
    protected_score = 5.0 if protected_area_distance_km <= 1.5 else 3.0 if protected_area_distance_km <= 5 else 1.0
    moisture_score = 5.0 if soil_moisture >= 0.35 else 3.0 if soil_moisture >= 0.25 else 1.0
    vegetation_score = 5.0 if ndvi >= 0.40 else 3.0 if ndvi >= 0.25 else 1.0
    buffer_score = 5.0 if water_buffer_m <= 300 else 3.0 if water_buffer_m <= 500 else 1.0

    return {
        "slope_score": round(slope_score, 2),
        "forest_score": round(forest_score, 2),
        "village_score": round(village_score, 2),
        "field_score": round(field_score, 2),
        "protected_score": round(protected_score, 2),
        "moisture_score": round(moisture_score, 2),
        "vegetation_score": round(vegetation_score, 2),
        "water_buffer_score": round(buffer_score, 2),
        "slope_pct": slope_pct,
        "ndvi": ndvi,
        "soil_moisture": soil_moisture,
        "distance_to_forest_m": forest_distance_m,
        "distance_to_village_km": village_distance_km,
        "distance_to_field_m": field_distance_m,
        "distance_to_protected_area_km": protected_area_distance_km,
        "water_buffer_m": water_buffer_m,
    }


def recommend_restoration_strategy(parcel: Dict[str, Any]) -> Dict[str, Any]:
    slope_pct = float(parcel.get("slope_pct", 0))
    ndvi = float(parcel.get("ndvi", 0.0))
    forest_distance_m = float(parcel.get("distance_to_forest_m", 9999))
    village_distance_km = float(parcel.get("distance_to_village_km", 9999))
    field_distance_m = float(parcel.get("distance_to_field_m", 9999))
    cropland = bool(parcel.get("cropland", False))

    if slope_pct > 35 or (ndvi < 0.25 and not cropland):
        strategy_id = "reboisement"
    elif 15 <= slope_pct <= 35 and forest_distance_m <= 800:
        strategy_id = "rna"
    elif slope_pct < 20 and cropland and village_distance_km <= 5.0:
        strategy_id = "agroforesterie"
    else:
        strategy_id = "baseline"

    details = SCENARIO_DETAILS[strategy_id]
    score = score_parcel(parcel)
    suitability = {
        "strategy_id": strategy_id,
        "strategy": details["title"],
        "summary": details["summary"],
        "benefits": details["benefits"],
        "risks": details["risks"],
        "score": score,
    }
    return suitability


def summarize_strategy(strategy: Dict[str, Any]) -> str:
    return f"{strategy['strategy']} — {strategy['summary']}"


if __name__ == "__main__":
    for parcel in build_sample_parcels():
        print(parcel["name"], recommend_restoration_strategy(parcel)["strategy"])
