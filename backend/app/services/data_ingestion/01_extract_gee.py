from __future__ import annotations

import csv
from pathlib import Path


OUTPUT_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "parcelles_nasa_metrics.csv"


def build_placeholder_dataset() -> list[dict]:
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
    ]


def main() -> None:
    rows = build_placeholder_dataset()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "id",
                "name",
                "slope_pct",
                "ndvi",
                "distance_to_forest_m",
                "distance_to_village_km",
                "distance_to_field_m",
                "distance_to_protected_area_km",
                "soil_moisture",
                "rainfall_mm",
                "temperature_c",
                "canopy_height_m",
                "water_buffer_m",
                "cropland",
                "degraded",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Created {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
