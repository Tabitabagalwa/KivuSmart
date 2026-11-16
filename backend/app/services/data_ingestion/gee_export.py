"""Téléchargement des produits NASA via Google Earth Engine (GEE).

Chaque produit est livré en GeoTIFF mono-bande, en UTM 35S (EPSG:32735), sur une
grille alignée (même origine pour une même résolution), ce qui permet ensuite
l'analyse de changement sans rééchantillonnage. Les pixels sans donnée valent -9999.

Prérequis (une seule fois) ::

    pip install earthengine-api
    earthengine authenticate          # ou ee.Authenticate() au premier lancement

Exemples (depuis la racine du projet) ::

    # Tous les produits, zone = parcelles pilotes
    python -m backend.app.services.data_ingestion.gee_export --project MON_PROJET_GCP

    # Seulement NDVI HLS pour 2016 et 2025, puis enregistrement dans l'API
    python -m backend.app.services.data_ingestion.gee_export --project MON_PROJET_GCP \
        --products hls_ndvi --years 2016 2025 --register --api-key VOTRE_CLE

    # Voir ce qui serait téléchargé, sans contacter GEE
    python -m backend.app.services.data_ingestion.gee_export --dry-run

Les fichiers trop volumineux pour un téléchargement direct (> 48 Mo) sont envoyés
vers Google Drive (dossier ``kivu_smart``) : récupérez-les ensuite dans ``data/raw/gee``.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import math
import os
import sys
import zipfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Callable

import httpx

from ...core.config import PROJECT_ROOT

logger = logging.getLogger("kivu.gee")

TARGET_CRS = "EPSG:32735"
NODATA = -9999.0
MAX_DIRECT_BYTES = 48 * 1024 * 1024  # limite des téléchargements directs GEE
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "raw" / "gee"
DEFAULT_AREA = PROJECT_ROOT / "data" / "boundaries" / "sud_kivu_pilotes.geojson"
DRIVE_FOLDER = "kivu_smart"
TIFF_MAGIC = (b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+")


# --------------------------------------------------------------------------- produits
def _hls_ndvi(ee, start: str, end: str):
    def ndvi(image):
        fmask = image.select("Fmask")
        clean = (
            fmask.bitwiseAnd(1 << 1).eq(0)       # nuage
            .And(fmask.bitwiseAnd(1 << 2).eq(0))  # adjacent nuage/ombre
            .And(fmask.bitwiseAnd(1 << 3).eq(0))  # ombre de nuage
        )
        return image.normalizedDifference(["B5", "B4"]).rename("ndvi").updateMask(clean)

    return ee.ImageCollection("NASA/HLS/HLSL30/v002").filterDate(start, end).map(ndvi).median()


def _modis_ndvi(ee, start: str, end: str):
    return (
        ee.ImageCollection("MODIS/061/MOD13Q1").filterDate(start, end)
        .select("NDVI").mean().multiply(0.0001).rename("ndvi")
    )


def _smap(ee, start: str, end: str):
    return (
        ee.ImageCollection("NASA/SMAP/SPL3SMP_E/006").filterDate(start, end)
        .select("soil_moisture_am").mean().rename("soil_moisture")
    )


def _gpm_rainfall(ee, start: str, end: str):
    # Taux moyen mensuel en mm/h converti en cumul annuel moyen (mm/an).
    return (
        ee.ImageCollection("NASA/GPM_L3/IMERG_MONTHLY_V07").filterDate(start, end)
        .select("precipitation").mean().multiply(24 * 365.25).rename("rainfall_mm")
    )


def _gedi(ee, start: str, end: str):
    return (
        ee.ImageCollection("LARSE/GEDI/GEDI02_A_002_MONTHLY").filterDate(start, end)
        .select("rh98").median().rename("canopy_height_m")
    )


def _nasadem(ee, _start: str, _end: str):
    return ee.Image("NASA/NASADEM_HGT/001").select("elevation")


def _landcover(ee, start: str, end: str):
    collection = ee.ImageCollection("MODIS/061/MCD12Q1").filterDate(start, end)
    return ee.Image(collection.first()).select("LC_Type1").rename("land_cover")


@dataclass(frozen=True)
class GeeProduct:
    key: str
    collection: str
    product: str
    source: str
    variable: str | None
    scale_m: int
    build: Callable[[Any, str, str], Any]
    mode: str  # "annual" (un fichier par année), "period" (une moyenne), "static"
    default_years: tuple[int, ...] = ()
    default_period: tuple[str, str] | None = None
    categorical: bool = False


PRODUCTS: dict[str, GeeProduct] = {
    p.key: p
    for p in (
        GeeProduct("nasadem", "NASA/NASADEM_HGT/001", "NASADEM_HGT v001 elevation",
                   "NASA LP DAAC via Google Earth Engine", "elevation_m", 30, _nasadem, "static",
                   default_period=("2000-02-11", "2000-02-22")),  # mission SRTM
        GeeProduct("hls_ndvi", "NASA/HLS/HLSL30/v002", "HLSL30 v002 NDVI (médiane annuelle, Fmask)",
                   "NASA LP DAAC via Google Earth Engine", "ndvi", 30, _hls_ndvi, "annual",
                   default_years=(2016, 2025)),
        GeeProduct("modis_ndvi", "MODIS/061/MOD13Q1", "MOD13Q1 v061 NDVI (moyenne annuelle)",
                   "NASA LP DAAC via Google Earth Engine", "ndvi", 250, _modis_ndvi, "annual",
                   default_years=(2016, 2025)),
        GeeProduct("smap", "NASA/SMAP/SPL3SMP_E/006", "SPL3SMP_E v006 soil_moisture_am (moyenne annuelle)",
                   "NASA NSIDC via Google Earth Engine", "soil_moisture", 9000, _smap, "annual",
                   default_years=(2024,)),
        GeeProduct("gpm_rainfall", "NASA/GPM_L3/IMERG_MONTHLY_V07", "GPM IMERG V07 pluie annuelle moyenne",
                   "NASA GES DISC via Google Earth Engine", "rainfall_mm", 11000, _gpm_rainfall, "period",
                   default_period=("2015-01-01", "2024-12-31")),
        GeeProduct("gedi", "LARSE/GEDI/GEDI02_A_002_MONTHLY", "GEDI L2A rh98 (médiane des empreintes)",
                   "NASA GEDI (LP DAAC) via Google Earth Engine", "canopy_height_m", 25, _gedi, "period",
                   default_period=("2019-04-01", "2023-03-31")),
        GeeProduct("modis_landcover", "MODIS/061/MCD12Q1", "MCD12Q1 v061 LC_Type1 (IGBP)",
                   "NASA LP DAAC via Google Earth Engine", "land_cover", 500, _landcover, "annual",
                   default_years=(2023,), categorical=True),
    )
}


@dataclass(frozen=True)
class Job:
    product: GeeProduct
    name: str
    start: str
    end: str


def plan_jobs(keys: list[str], years: list[int] | None = None,
              period: tuple[str, str] | None = None) -> list[Job]:
    unknown = sorted(set(keys) - set(PRODUCTS))
    if unknown:
        raise ValueError(f"Produits inconnus : {unknown}. Disponibles : {sorted(PRODUCTS)}")
    jobs: list[Job] = []
    for key in keys:
        product = PRODUCTS[key]
        if product.mode == "annual":
            for year in years or product.default_years:
                jobs.append(Job(product, f"{key}_{year}", f"{year}-01-01", f"{year}-12-31"))
        else:
            start, end = (period if period and product.mode == "period" else product.default_period)
            suffix = "" if product.mode == "static" else f"_{start[:4]}_{end[:4]}"
            jobs.append(Job(product, f"{key}{suffix}", start, end))
    return jobs


# ------------------------------------------------------------------------ zone & grille
def load_area_bounds(path: Path, buffer_deg: float = 0.01) -> tuple[float, float, float, float]:
    """Emprise WGS 84 (ouest, sud, est, nord) du fichier de zone, avec marge."""
    import geopandas as gpd

    frame = gpd.read_file(path)
    if frame.empty or frame.crs is None:
        raise ValueError(f"Zone vide ou sans système de coordonnées : {path}")
    west, south, east, north = frame.to_crs("EPSG:4326").total_bounds
    return (west - buffer_deg, south - buffer_deg, east + buffer_deg, north + buffer_deg)


def utm_grid(bounds: tuple[float, float, float, float], scale_m: float) -> dict[str, Any]:
    """Grille UTM 35S alignée sur des multiples de ``scale_m`` couvrant ``bounds``."""
    from pyproj import Transformer

    west, south, east, north = bounds
    transformer = Transformer.from_crs("EPSG:4326", TARGET_CRS, always_xy=True)
    xs, ys = transformer.transform([west, east, west, east], [south, south, north, north])
    left = math.floor(min(xs) / scale_m) * scale_m
    right = math.ceil(max(xs) / scale_m) * scale_m
    bottom = math.floor(min(ys) / scale_m) * scale_m
    top = math.ceil(max(ys) / scale_m) * scale_m
    width = max(1, int(round((right - left) / scale_m)))
    height = max(1, int(round((top - bottom) / scale_m)))
    return {
        "crs": TARGET_CRS,
        "crs_transform": [scale_m, 0, left, 0, -scale_m, top],
        "width": width,
        "height": height,
        "estimated_bytes": width * height * 4,
    }


# ------------------------------------------------------------------------ téléchargement
def _extract_tiff(content: bytes) -> bytes:
    if content[:4] in TIFF_MAGIC:
        return content
    if content[:2] == b"PK":  # certaines réponses GEE sont zippées
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            tiffs = [n for n in archive.namelist() if n.lower().endswith((".tif", ".tiff"))]
            if len(tiffs) != 1:
                raise ValueError(f"Archive GEE inattendue : {archive.namelist()}")
            return archive.read(tiffs[0])
    snippet = content[:200].decode("utf-8", errors="replace")
    raise ValueError(f"Réponse GEE qui n'est pas un GeoTIFF : {snippet}")


def finalize_tiff(path: Path, job: Job) -> None:
    """Déclare le nodata et inscrit la provenance dans les métadonnées du GeoTIFF."""
    import rasterio

    with rasterio.open(path, "r+") as dataset:
        if dataset.count != 1:
            raise ValueError(f"{path.name} contient {dataset.count} bandes au lieu d'une.")
        if dataset.crs is None:
            raise ValueError(f"{path.name} n'a pas de système de coordonnées.")
        dataset.nodata = NODATA
        dataset.set_band_description(1, job.product.variable or job.product.key)
        dataset.update_tags(
            source=job.product.source,
            product=job.product.product,
            gee_collection=job.product.collection,
            period_start=job.start,
            period_end=job.end,
            satellite="true",
            native_scale_m=str(job.product.scale_m),
            resolution_note="Grille d'export GEE ; aucune information n'est ajoutée par le rééchantillonnage.",
        )


def download_job(ee, job: Job, bounds, output_dir: Path, *, force_drive: bool = False,
                 http: httpx.Client | None = None) -> dict[str, Any]:
    grid = utm_grid(bounds, job.product.scale_m)
    image = job.product.build(ee, job.start, job.end).toFloat().unmask(NODATA, False)
    record: dict[str, Any] = {"name": job.name, "grid": grid}

    if force_drive or grid["estimated_bytes"] > MAX_DIRECT_BYTES:
        task = ee.batch.Export.image.toDrive(
            image=image, description=job.name, folder=DRIVE_FOLDER, fileNamePrefix=job.name,
            crs=grid["crs"], crsTransform=grid["crs_transform"],
            dimensions=f"{grid['width']}x{grid['height']}", fileFormat="GeoTIFF", maxPixels=1e10,
        )
        task.start()
        record.update(status="drive_export_started", task_id=task.id,
                      message=f"Récupérez {job.name}.tif dans Google Drive/{DRIVE_FOLDER}, puis copiez-le dans {output_dir}.")
        return record

    url = image.getDownloadURL({
        "name": job.name,
        "format": "GEO_TIFF",
        "crs": grid["crs"],
        "crs_transform": grid["crs_transform"],
        "dimensions": f"{grid['width']}x{grid['height']}",
    })
    client = http or httpx.Client(timeout=300, follow_redirects=True)
    try:
        response = client.get(url)
        response.raise_for_status()
    finally:
        if http is None:
            client.close()
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"{job.name}.tif"
    temporary = destination.with_suffix(".part")
    temporary.write_bytes(_extract_tiff(response.content))
    try:
        finalize_tiff(temporary, job)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    temporary.replace(destination)
    record.update(status="downloaded", path=str(destination))
    return record


# ------------------------------------------------------------------- enregistrement API
def register_in_api(path: Path, job: Job, api_url: str, api_key: str | None,
                    http: httpx.Client | None = None) -> dict[str, Any]:
    data = {
        "name": job.name,
        "source": job.product.source,
        "product": job.product.product,
        "satellite": "true",
        "categorical": str(job.product.categorical).lower(),
        "period_start": job.start,
        "period_end": job.end,
    }
    if job.product.variable and not job.product.categorical:
        data["variable"] = job.product.variable
    headers = {"X-API-Key": api_key} if api_key else {}
    client = http or httpx.Client(timeout=300)
    try:
        with path.open("rb") as handle:
            response = client.post(f"{api_url.rstrip('/')}/api/v1/datasets/raster", headers=headers,
                                   data=data, files={"file": (path.name, handle, "image/tiff")})
    finally:
        if http is None:
            client.close()
    if response.status_code != 201:
        return {"registered": False, "status_code": response.status_code, "detail": response.text[:300]}
    return {"registered": True, "dataset_id": response.json()["id"]}


# ---------------------------------------------------------------------------------- CLI
def _initialize_ee(project: str):
    try:
        import ee
    except ImportError as exc:
        raise SystemExit("Installez d'abord earthengine-api : pip install earthengine-api") from exc
    try:
        ee.Initialize(project=project)
    except Exception as first_error:
        message = str(first_error).lower()
        if "not signed up" in message or "not registered" in message or "permission" in message:
            # Les identifiants sont valides : c'est le projet qui pose problème.
            raise SystemExit(
                f"Le projet Google Cloud {project!r} n'est pas enregistré pour Earth Engine "
                "(ou votre compte n'y a pas accès).\n"
                "Enregistrez-le sur https://console.cloud.google.com/earth-engine puis relancez.\n"
                f"Détail : {first_error}"
            ) from first_error
        ee.Authenticate()  # identifiants absents ou expirés : ouvre le navigateur
        ee.Initialize(project=project)
    try:  # vérifie tout de suite que le projet répond, avant de lancer les téléchargements
        ee.Image("NASA/NASADEM_HGT/001").bandNames().getInfo()
    except Exception as exc:
        raise SystemExit(f"Earth Engine ne répond pas avec le projet {project!r} : {exc}") from exc
    return ee


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Téléchargement des produits NASA via Google Earth Engine.")
    parser.add_argument("--project", help="Identifiant du projet Google Cloud lié à Earth Engine")
    parser.add_argument("--area", type=Path, default=DEFAULT_AREA,
                        help="Fichier vectoriel de la zone (GeoJSON, GPKG, Shapefile)")
    parser.add_argument("--products", nargs="+", default=list(PRODUCTS), choices=sorted(PRODUCTS))
    parser.add_argument("--years", nargs="+", type=int, help="Années pour les produits annuels")
    parser.add_argument("--period", nargs=2, metavar=("DEBUT", "FIN"),
                        help="Période AAAA-MM-JJ pour gpm_rainfall et gedi")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--drive", action="store_true", help="Toujours exporter vers Google Drive")
    parser.add_argument("--register", action="store_true", help="Enregistrer les fichiers dans l'API Kivu Smart")
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--api-key", default=os.getenv("KIVU_API_KEY"))
    parser.add_argument("--dry-run", action="store_true", help="Afficher le plan sans contacter GEE")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.years and any(y < 2000 or y > date.today().year for y in args.years):
        parser.error("Années hors limites (2000 à l'année courante).")
    if args.period:
        for value in args.period:
            date.fromisoformat(value)
    jobs = plan_jobs(args.products, args.years, tuple(args.period) if args.period else None)
    bounds = load_area_bounds(args.area)
    if args.area.resolve() == DEFAULT_AREA.resolve():
        logger.warning("Zone = parcelles pilotes de démonstration. Utilisez --area pour la limite officielle de Kalehe.")

    if args.dry_run:
        for job in jobs:
            grid = utm_grid(bounds, job.product.scale_m)
            mode = "Drive" if grid["estimated_bytes"] > MAX_DIRECT_BYTES or args.drive else "direct"
            logger.info("%-28s %s  %s→%s  %dx%d px @ %d m  (%s)", job.name, job.product.collection,
                        job.start, job.end, grid["width"], grid["height"], job.product.scale_m, mode)
        return 0
    if not args.project:
        parser.error("--project est obligatoire (sauf avec --dry-run).")

    ee = _initialize_ee(args.project)
    results, failures = [], 0
    for job in jobs:
        logger.info("→ %s", job.name)
        try:
            record = download_job(ee, job, bounds, args.output, force_drive=args.drive)
            if args.register and record["status"] == "downloaded":
                record["api"] = register_in_api(Path(record["path"]), job, args.api_url, args.api_key)
        except Exception as exc:  # un produit en échec n'arrête pas les autres
            failures += 1
            record = {"name": job.name, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        logger.info("   %s", record.get("message") or record["status"])
        results.append({**record, "collection": job.product.collection, "product": job.product.product,
                        "source": job.product.source, "period_start": job.start, "period_end": job.end,
                        "variable": job.product.variable, "categorical": job.product.categorical})

    args.output.mkdir(parents=True, exist_ok=True)
    manifest = args.output / "gee_manifest.json"
    manifest.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Manifeste : %s (%d produit(s) en échec)", manifest, failures)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
