"""NASA POWER (Prediction Of Worldwide Energy Resources) point API.

No credentials are needed. POWER values are model/satellite-derived grids
(about 0.5° x 0.625° for MERRA-2 meteorology), so one point represents a large
cell, not a parcel: values are context for the decision, not field measurements.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import date
from typing import Any, Literal

from .client import NasaHttpClient
from .errors import NasaApiError

POWER_BASE_URL = "https://power.larc.nasa.gov/api/temporal"
SERVICE = "power"
MAX_PARAMETERS = 15
FIRST_YEAR = 1981

Temporal = Literal["daily", "monthly", "climatology"]

# Agro-climatic parameters relevant to restoration in South Kivu.
DEFAULT_PARAMETERS = (
    "T2M",          # air temperature at 2 m (°C)
    "T2M_MAX",
    "T2M_MIN",
    "PRECTOTCORR",  # bias-corrected precipitation (mm/day)
    "RH2M",         # relative humidity at 2 m (%)
    "ALLSKY_SFC_SW_DWN",  # surface shortwave irradiance
    "GWETROOT",     # root-zone soil wetness (0-1)
)

_PARAMETER_PATTERN = re.compile(r"^[A-Z0-9_]{2,40}$")
_MONTH_KEYS = {f"{m:02d}" for m in range(1, 13)}


@dataclass(frozen=True)
class PowerRequest:
    latitude: float
    longitude: float
    start: date
    end: date
    temporal: Temporal = "monthly"
    parameters: tuple[str, ...] = DEFAULT_PARAMETERS
    community: str = "AG"

    def validate(self) -> None:
        if not (-90 <= self.latitude <= 90) or not (-180 <= self.longitude <= 180):
            raise ValueError("Latitude/longitude hors limites.")
        if not self.parameters or len(self.parameters) > MAX_PARAMETERS:
            raise ValueError(f"Entre 1 et {MAX_PARAMETERS} paramètres POWER sont requis.")
        invalid = [p for p in self.parameters if not _PARAMETER_PATTERN.match(p)]
        if invalid:
            raise ValueError(f"Paramètres POWER invalides : {invalid}")
        if self.community not in {"AG", "RE", "SB"}:
            raise ValueError("Communauté POWER invalide (AG, RE ou SB).")
        if self.start > self.end:
            raise ValueError("La date de début est postérieure à la date de fin.")
        if self.start.year < FIRST_YEAR:
            raise ValueError(f"POWER commence en {FIRST_YEAR}.")
        if self.end > date.today():
            raise ValueError("La date de fin ne peut pas être dans le futur.")

    def query(self) -> dict[str, Any]:
        params: dict[str, Any] = {
            "parameters": ",".join(self.parameters),
            "community": self.community,
            "latitude": round(self.latitude, 4),
            "longitude": round(self.longitude, 4),
            "format": "JSON",
        }
        if self.temporal == "daily":
            params.update(start=self.start.strftime("%Y%m%d"), end=self.end.strftime("%Y%m%d"))
        else:
            params.update(start=self.start.year, end=self.end.year)
        return params


def fetch_point(client: NasaHttpClient, request: PowerRequest) -> dict[str, Any]:
    request.validate()
    url = f"{POWER_BASE_URL}/{request.temporal}/point"
    payload = client.get_json(url, params=request.query())
    return parse_power_response(payload, request.temporal)


def parse_power_response(payload: dict[str, Any], temporal: Temporal) -> dict[str, Any]:
    try:
        parameter_block: dict[str, dict[str, Any]] = payload["properties"]["parameter"]
    except (KeyError, TypeError) as exc:
        messages = payload.get("messages") if isinstance(payload, dict) else None
        raise NasaApiError(SERVICE, f"Réponse POWER inattendue : {messages or 'champ parameter absent'}") from exc

    header = payload.get("header", {}) if isinstance(payload, dict) else {}
    fill_value = header.get("fill_value", -999.0)
    units = {
        name: (info or {}).get("units")
        for name, info in (payload.get("parameters") or {}).items()
    }

    series: dict[str, list[dict[str, Any]]] = {}
    annual: dict[str, dict[str, float | None]] = {}
    for parameter, values in parameter_block.items():
        records: list[dict[str, Any]] = []
        for key, raw in values.items():
            value = _clean(raw, fill_value)
            if temporal == "monthly" and key.endswith("13") and len(key) == 6:
                annual.setdefault(parameter, {})[key[:4]] = value  # POWER annual aggregate
                continue
            if temporal == "climatology" and key not in _MONTH_KEYS:
                annual.setdefault(parameter, {})[key] = value  # "ANN"
                continue
            records.append({"period": key, "value": value})
        series[parameter] = sorted(records, key=lambda r: r["period"])

    return {
        "temporal": temporal,
        "coordinates": (payload.get("geometry") or {}).get("coordinates"),
        "units": units,
        "fill_value": fill_value,
        "series": series,
        "annual": annual,
        "summary": summarize_series(series),
        "source": "NASA POWER (power.larc.nasa.gov)",
        "resolution_note": (
            "Grille POWER ~0,5° x 0,625° : valeur régionale, à valider localement."
        ),
    }


def _clean(raw: Any, fill_value: Any) -> float | None:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value) or value == float(fill_value):
        return None
    return value


def summarize_series(series: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for parameter, records in series.items():
        values = [r["value"] for r in records if r["value"] is not None]
        summary[parameter] = {
            "count": len(values),
            "missing": len(records) - len(values),
            "mean": sum(values) / len(values) if values else None,
            "min": min(values) if values else None,
            "max": max(values) if values else None,
        }
    return summary


def to_parcel_metrics(result: dict[str, Any]) -> dict[str, tuple[float, str]]:
    """Map POWER summaries to the decision-engine variables (value, unit)."""
    summary = result["summary"]
    metrics: dict[str, tuple[float, str]] = {}
    precipitation = summary.get("PRECTOTCORR", {}).get("mean")
    if precipitation is not None:
        metrics["rainfall_mm"] = (round(precipitation * 365.25, 1), "mm/an")
    temperature = summary.get("T2M", {}).get("mean")
    if temperature is not None:
        metrics["temperature_c"] = (round(temperature, 2), "°C")
    t_max = summary.get("T2M_MAX", {}).get("mean")
    if t_max is not None:
        metrics["temperature_max_c"] = (round(t_max, 2), "°C")
    root_wetness = summary.get("GWETROOT", {}).get("mean")
    if root_wetness is not None:
        metrics["root_zone_wetness"] = (round(root_wetness, 3), "0-1")
    return metrics
