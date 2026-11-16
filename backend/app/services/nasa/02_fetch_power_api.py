from __future__ import annotations

import os
from typing import Any, Dict

import requests


NASA_POWER_BASE_URL = "https://power.larc.nasa.gov/api/temporal/monthly/point"


def fetch_power_metrics(latitude: float, longitude: float, parameters: list[str] | None = None) -> Dict[str, Any]:
    parameters = parameters or ["T2M", "PRECTOTCORR", "RH2M", "ALLSKY_SFC_SW_DWN"]
    payload = {
        "parameters": ",".join(parameters),
        "community": "RE",
        "longitude": longitude,
        "latitude": latitude,
        "start": 2024,
        "end": 2025,
        "format": "JSON",
    }
    response = requests.get(NASA_POWER_BASE_URL, params=payload, timeout=30)
    response.raise_for_status()
    return response.json()


def main() -> None:
    latitude = float(os.getenv("KIVU_LATITUDE", "-2.33"))
    longitude = float(os.getenv("KIVU_LONGITUDE", "28.79"))
    try:
        metrics = fetch_power_metrics(latitude, longitude)
        print(metrics["properties"]["parameter"])
    except Exception as exc:  # pragma: no cover - allows graceful demo behavior without external network access
        print(f"NASA POWER request failed: {exc}")
        print("Check your internet connection or the POWER API availability.")


if __name__ == "__main__":
    main()
