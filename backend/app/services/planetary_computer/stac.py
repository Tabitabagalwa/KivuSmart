"""Recherche STAC et lecture de COG sur Microsoft Planetary Computer.

Les archives NASA / USGS (NASADEM, Landsat Collection 2, MODIS) y sont publiées en
Cloud-Optimized GeoTIFF : seules les tuiles nécessaires sont lues, directement
reprojetées sur la grille du projet. Les URL sont signées avec un jeton SAS public
(aucun compte n'est requis).
"""

from __future__ import annotations

import time
from typing import Any
from urllib.parse import urlsplit

import httpx
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT

STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
SAS = "https://planetarycomputer.microsoft.com/api/sas/v1/token"
HOTES = {"planetarycomputer.microsoft.com"}
SUFFIXE_STOCKAGE = ".blob.core.windows.net"
USER_AGENT = "KivuSmartRestoration/2.0 (NASA Space Apps research prototype)"

GDAL_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.TIF,.tiff",
    "GDAL_HTTP_MAX_RETRY": "4",
    "GDAL_HTTP_RETRY_DELAY": "2",
    "GDAL_HTTP_MULTIRANGE": "YES",
    "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
    "VSI_CACHE": "TRUE",
}


class PlanetaryComputerErreur(RuntimeError):
    pass


def _verifier(url: str, stockage: bool = False) -> None:
    morceaux = urlsplit(url)
    hote = morceaux.hostname or ""
    if morceaux.scheme != "https" or not (hote in HOTES or (stockage and hote.endswith(SUFFIXE_STOCKAGE))):
        raise PlanetaryComputerErreur(f"Hôte non autorisé : {hote}")


class Catalogue:
    def __init__(self, timeout_s: float = 90.0) -> None:
        self.http = httpx.Client(timeout=httpx.Timeout(timeout_s, connect=20.0),
                                 headers={"User-Agent": USER_AGENT})
        self._jetons: dict[str, tuple[str, float]] = {}

    def close(self) -> None:
        self.http.close()

    def __enter__(self) -> "Catalogue":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def rechercher(self, collection: str, bbox, datetime: str, query: dict | None = None,
                   limite: int = 500) -> list[dict[str, Any]]:
        corps: dict[str, Any] = {"collections": [collection], "bbox": list(bbox), "datetime": datetime,
                                 "limit": min(limite, 250)}
        if query:
            corps["query"] = query
        url, items = f"{STAC}/search", []
        while url and len(items) < limite:
            _verifier(url)
            reponse = self.http.post(url, json=corps)
            reponse.raise_for_status()
            page = reponse.json()
            items += page.get("features", [])
            suivant = next((l for l in page.get("links", []) if l.get("rel") == "next"), None)
            if not suivant:
                break
            url, corps = suivant["href"], suivant.get("body", corps)
        return items[:limite]

    def jeton(self, collection: str) -> str:
        jeton, expiration = self._jetons.get(collection, ("", 0.0))
        if time.time() < expiration - 300:
            return jeton
        url = f"{SAS}/{collection}"
        _verifier(url)
        reponse = self.http.get(url)
        reponse.raise_for_status()
        jeton = reponse.json()["token"]
        self._jetons[collection] = (jeton, time.time() + 45 * 60)
        return jeton

    def href_signe(self, item: dict[str, Any], actif: str) -> str:
        href = item["assets"][actif]["href"]
        _verifier(href, stockage=True)
        return f"{href}?{self.jeton(item['collection'])}"


def lire_sur_grille(href: str, crs: str, transform, largeur: int, hauteur: int,
                    resampling: Resampling = Resampling.bilinear, dtype: str = "float32",
                    nodata_source: float | None = None) -> np.ndarray:
    """Lit un COG reprojeté sur la grille cible ; NaN hors couverture ou sans donnée."""
    with rasterio.Env(**GDAL_ENV):
        with rasterio.open(f"/vsicurl/{href}" if href.startswith("http") else href) as source:
            nodata = source.nodata if nodata_source is None else nodata_source
            with WarpedVRT(source, crs=crs, transform=transform, width=largeur, height=hauteur,
                           resampling=resampling, src_nodata=nodata, nodata=nodata,
                           add_alpha=False) as vrt:
                valeurs = vrt.read(1, masked=True)
    sortie = valeurs.astype(dtype).filled(np.nan) if np.issubdtype(np.dtype(dtype), np.floating) \
        else valeurs.filled(0).astype(dtype)
    return sortie
