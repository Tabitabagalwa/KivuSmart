"""NASA Common Metadata Repository (CMR) granule search.

Searching is public. Downloading the returned granule links requires an Earthdata
Login token (``KIVU_EARTHDATA_TOKEN``).
"""

from __future__ import annotations

from datetime import date
from typing import Any

from .catalog import NasaProduct
from .client import NasaHttpClient
from .errors import NasaApiError

CMR_GRANULES_URL = "https://cmr.earthdata.nasa.gov/search/granules.json"
SERVICE = "cmr"
DATA_REL_SUFFIX = "/data#"
MAX_PAGE_SIZE = 200


def _validate_bbox(bbox: tuple[float, float, float, float]) -> None:
    west, south, east, north = bbox
    if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
        raise ValueError("bbox invalide : attendu ouest,sud,est,nord en degrés WGS 84.")


def search_granules(
    client: NasaHttpClient,
    product: NasaProduct,
    bbox: tuple[float, float, float, float],
    start: date,
    end: date,
    *,
    page_size: int = 50,
    version: str | None = None,
    max_cloud_cover: float | None = None,
) -> dict[str, Any]:
    _validate_bbox(bbox)
    if start > end:
        raise ValueError("La date de début est postérieure à la date de fin.")
    if "sentinel" in product.short_name.casefold() or product.short_name.upper() == "HLSS30":
        raise ValueError("Les produits Sentinel ne sont pas autorisés dans ce projet.")
    params: dict[str, Any] = {
        "short_name": product.short_name,
        "bounding_box": ",".join(f"{v:.6f}" for v in bbox),
        "temporal": f"{start.isoformat()}T00:00:00Z,{end.isoformat()}T23:59:59Z",
        "page_size": max(1, min(page_size, MAX_PAGE_SIZE)),
        "sort_key": "-start_date",
    }
    chosen_version = version or product.version
    if chosen_version:
        params["version"] = chosen_version
    if max_cloud_cover is not None:
        params["cloud_cover"] = f"0,{max_cloud_cover:g}"

    payload = client.get_json(CMR_GRANULES_URL, params=params)
    try:
        entries = payload["feed"]["entry"]
    except (KeyError, TypeError) as exc:
        raise NasaApiError(SERVICE, "Réponse CMR inattendue (feed.entry absent).") from exc

    granules = [_granule(entry) for entry in entries]
    return {
        "product": product.as_dict(),
        "query": params,
        "count": len(granules),
        "granules": granules,
        "download_requires": "Jeton Earthdata Login (KIVU_EARTHDATA_TOKEN)",
    }


def _granule(entry: dict[str, Any]) -> dict[str, Any]:
    links = entry.get("links") or []
    data_links = [
        link.get("href")
        for link in links
        if str(link.get("rel", "")).endswith(DATA_REL_SUFFIX)
        and str(link.get("href", "")).startswith("https://")
        and not link.get("inherited")
    ]
    cloud = entry.get("cloud_cover")
    return {
        "id": entry.get("id"),
        "title": entry.get("title"),
        "time_start": entry.get("time_start"),
        "time_end": entry.get("time_end"),
        "cloud_cover": float(cloud) if cloud not in (None, "") else None,
        "size_mb": float(entry["granule_size"]) if entry.get("granule_size") else None,
        "data_links": data_links,
    }
