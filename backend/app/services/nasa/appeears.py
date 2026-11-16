"""NASA AppEEARS (LP DAAC) area/point extraction API.

AppEEARS clips NASA products to an area and returns GeoTIFFs, which suits the
existing harmonization pipeline. It requires an Earthdata Login account
(``KIVU_EARTHDATA_USERNAME`` / ``KIVU_EARTHDATA_PASSWORD``); the session token
returned by ``/login`` is reused for subsequent calls.

Product and layer names change over time: check them with :func:`product_layers`.
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path
from typing import Any, Literal

from .client import NasaHttpClient
from .errors import NasaApiError, NasaConfigurationError

APPEEARS_BASE_URL = "https://appeears.earthdatacloud.nasa.gov/api"
SERVICE = "appeears"
_TASK_ID = re.compile(r"^[0-9a-fA-F-]{8,64}$")


def _check_id(value: str, label: str) -> str:
    if not _TASK_ID.match(value):
        raise ValueError(f"Identifiant {label} AppEEARS invalide.")
    return value


def login(client: NasaHttpClient, username: str | None, password: str | None) -> str:
    if not username or not password:
        raise NasaConfigurationError(
            SERVICE,
            "Identifiants Earthdata absents : définissez KIVU_EARTHDATA_USERNAME et "
            "KIVU_EARTHDATA_PASSWORD.",
        )
    payload = client.post_json(f"{APPEEARS_BASE_URL}/login", None, basic_auth=(username, password))
    token = payload.get("token") if isinstance(payload, dict) else None
    if not token:
        raise NasaApiError(SERVICE, "Jeton AppEEARS absent de la réponse /login.")
    client.bearer_token = token
    return token


def product_layers(client: NasaHttpClient, product: str) -> dict[str, Any]:
    if not re.match(r"^[A-Za-z0-9_.]+$", product):
        raise ValueError("Nom de produit AppEEARS invalide.")
    return client.get_json(f"{APPEEARS_BASE_URL}/product/{product}")


def build_area_task(
    task_name: str,
    geojson: dict[str, Any],
    layers: list[dict[str, str]],
    start: date,
    end: date,
    output_format: Literal["geotiff", "netcdf4"] = "geotiff",
) -> dict[str, Any]:
    if start > end:
        raise ValueError("La date de début est postérieure à la date de fin.")
    if not layers:
        raise ValueError("Au moins une couche AppEEARS est requise.")
    for layer in layers:
        if "sentinel" in f"{layer.get('product', '')}{layer.get('layer', '')}".casefold():
            raise ValueError("Les produits Sentinel ne sont pas autorisés dans ce projet.")
    if geojson.get("type") != "FeatureCollection" or not geojson.get("features"):
        raise ValueError("La zone doit être une FeatureCollection GeoJSON non vide.")
    return {
        "task_type": "area",
        "task_name": re.sub(r"[^A-Za-z0-9_-]+", "_", task_name)[:80] or "kivu_task",
        "params": {
            "dates": [{"startDate": start.strftime("%m-%d-%Y"), "endDate": end.strftime("%m-%d-%Y")}],
            "layers": [{"product": l["product"], "layer": l["layer"]} for l in layers],
            "geo": geojson,
            "output": {"format": {"type": output_format}, "projection": "geographic"},
        },
    }


def submit_task(client: NasaHttpClient, task: dict[str, Any]) -> dict[str, Any]:
    payload = client.post_json(f"{APPEEARS_BASE_URL}/task", task, auth=True)
    if not isinstance(payload, dict) or "task_id" not in payload:
        raise NasaApiError(SERVICE, "Réponse de soumission AppEEARS inattendue.")
    return payload


def task_status(client: NasaHttpClient, task_id: str) -> dict[str, Any]:
    return client.get_json(
        f"{APPEEARS_BASE_URL}/task/{_check_id(task_id, 'tâche')}", auth=True, use_cache=False
    )


def bundle(client: NasaHttpClient, task_id: str) -> dict[str, Any]:
    return client.get_json(
        f"{APPEEARS_BASE_URL}/bundle/{_check_id(task_id, 'tâche')}", auth=True, use_cache=False
    )


def download_bundle_file(
    client: NasaHttpClient,
    task_id: str,
    file_id: str,
    file_name: str,
    destination_dir: Path,
    max_bytes: int,
) -> Path:
    safe_name = Path(file_name).name
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", safe_name).strip("._") or f"{file_id}.bin"
    url = (
        f"{APPEEARS_BASE_URL}/bundle/{_check_id(task_id, 'tâche')}/"
        f"{_check_id(file_id, 'fichier')}"
    )
    return client.download(url, destination_dir / task_id / safe_name, max_bytes=max_bytes, auth=True)
