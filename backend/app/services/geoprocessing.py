"""Geospatial processing service: bridges ``backend.app.processing`` and the database.

Raster/vector algorithms stay in the processing package; this module selects
inputs from registered datasets, runs them and stores parcel observations with
their provenance and quality indicators.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
from shapely.ops import unary_union
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..db.models import Dataset, Parcel
from ..db.repositories import get_dataset
from ..processing.errors import GeospatialProcessingError
from ..processing.features import extract_zonal_statistics
from ..processing.indices.vegetation import calculate_ndvi
from ..processing.pipeline import run_configured_pipeline
from ..processing.raster.terrain import derive_terrain_attributes
from .data_ingestion.importers import (
    METRIC_CRS,
    STORAGE_CRS,
    add_observation,
    dataset_path,
    read_vector,
    register_raster,
)

# Default mapping from raster variable to stored parcel variable.
RASTER_VARIABLES = {
    "slope_percent": "slope_pct",
    "slope": "slope_pct",
    "ndvi": "ndvi",
    "soil_moisture": "soil_moisture",
    "canopy_height": "canopy_height_m",
    "elevation": "elevation_m",
    "land_surface_temperature": "land_surface_temperature_c",
}


def parcels_frame(session: Session, codes: list[str] | None = None) -> gpd.GeoDataFrame:
    statement = select(Parcel).order_by(Parcel.code)
    if codes:
        statement = statement.where(Parcel.code.in_(codes))
    parcels = list(session.scalars(statement))
    if not parcels:
        raise GeospatialProcessingError("Aucune parcelle à traiter.")
    if codes:
        missing = sorted(set(codes) - {p.code for p in parcels})
        if missing:
            raise GeospatialProcessingError(f"Parcelles inconnues : {missing}")
    return gpd.GeoDataFrame(
        {"code": [p.code for p in parcels], "parcel_id": [p.id for p in parcels],
         "is_demo": [p.is_demo for p in parcels]},
        geometry=[p.geometry for p in parcels],
        crs=STORAGE_CRS,
    )


def zonal_observations(
    session: Session,
    data_dir: Path,
    dataset_ids: list[int],
    parcel_codes: list[str] | None = None,
) -> dict[str, Any]:
    """Compute mean raster values per parcel and store them as observations."""
    zones = parcels_frame(session, parcel_codes)
    stored: list[dict[str, Any]] = []
    errors: list[str] = []
    for dataset_id in dataset_ids:
        dataset = get_dataset(session, dataset_id)
        if dataset.kind != "raster":
            errors.append(f"{dataset.name} : n'est pas un raster.")
            continue
        if dataset.categorical:
            errors.append(
                f"{dataset.name} est catégoriel : une moyenne n'a pas de sens, aucune valeur stockée."
            )
            continue
        variable = dataset.variable or RASTER_VARIABLES.get(dataset.name, dataset.name)
        variable = RASTER_VARIABLES.get(variable, variable)
        path = dataset_path(dataset, data_dir)
        for _, zone in zones.iterrows():
            single = zones.loc[[zone.name]]
            try:
                stats = extract_zonal_statistics({dataset.name: path}, single, zone_id_field="code")
            except GeospatialProcessingError as exc:
                errors.append(f"{zone.code} / {dataset.name} : {exc}")
                continue
            row = stats.iloc[0]
            parcel = session.get(Parcel, int(zone.parcel_id))
            add_observation(
                session,
                parcel,
                variable,
                float(row["mean"]),
                source=f"{dataset.source} — {dataset.product}",
                method="zonal_mean",
                observed_start=dataset.period_start,
                observed_end=dataset.period_end,
                dataset_id=dataset.id,
                is_demo=bool(zone.is_demo),
                quality={
                    "valid_pixel_count": int(row["valid_pixel_count"]),
                    "std": float(row["std"]),
                    "min": float(row["min"]),
                    "max": float(row["max"]),
                    "raster_resolution": [float(row["resolution_x"]), float(row["resolution_y"])],
                    "raster_crs": row["crs"],
                    "note": (
                        "Un faible nombre de pixels valides signifie que la parcelle est plus "
                        "petite que la résolution du capteur."
                    ),
                },
            )
            stored.append({"parcel": zone.code, "variable": variable, "value": float(row["mean"]),
                           "valid_pixel_count": int(row["valid_pixel_count"])})
    session.flush()
    return {"stored": stored, "errors": errors}


def proximity_observations(
    session: Session,
    data_dir: Path,
    dataset_id: int,
    variable: str,
    parcel_codes: list[str] | None = None,
) -> dict[str, Any]:
    """Distance from each parcel edge to the nearest feature, computed in UTM 35S."""
    if not variable.startswith("distance_to_") and variable != "water_buffer_m":
        raise GeospatialProcessingError(
            "La variable de proximité doit commencer par 'distance_to_' ou être 'water_buffer_m'."
        )
    dataset = get_dataset(session, dataset_id)
    if dataset.kind != "vector":
        raise GeospatialProcessingError(f"{dataset.name} n'est pas une couche vectorielle.")
    features = read_vector(dataset_path(dataset, data_dir)).to_crs(METRIC_CRS)
    features = features[features.geometry.notna() & ~features.geometry.is_empty]
    if features.empty:
        raise GeospatialProcessingError("La couche de référence ne contient aucune géométrie.")
    target = unary_union(list(features.geometry))
    zones = parcels_frame(session, parcel_codes).to_crs(METRIC_CRS)
    factor = 0.001 if variable.endswith("_km") else 1.0
    stored = []
    for _, zone in zones.iterrows():
        distance_m = float(zone.geometry.distance(target))
        value = round(distance_m * factor, 4)
        add_observation(
            session,
            session.get(Parcel, int(zone.parcel_id)),
            variable,
            value,
            source=f"{dataset.source} — {dataset.product}",
            method="proximity_utm35s",
            dataset_id=dataset.id,
            is_demo=bool(zone.is_demo),
            quality={"crs": METRIC_CRS, "distance_from": "bord de la parcelle"},
        )
        stored.append({"parcel": zone.code, "variable": variable, "value": value})
    session.flush()
    return {"stored": stored}


def derive_terrain(session: Session, data_dir: Path, dataset_id: int) -> list[Dataset]:
    """Derive slope (%) and aspect from a registered, projected DEM."""
    dem = get_dataset(session, dataset_id)
    if dem.kind != "raster":
        raise GeospatialProcessingError("Le MNT doit être un raster.")
    output_dir = data_dir / "processed" / "terrain" / f"dataset_{dem.id}"
    outputs = derive_terrain_attributes(dataset_path(dem, data_dir), output_dir)
    created = []
    for key, path in outputs.items():
        created.append(
            register_raster(
                session,
                Path(path),
                data_dir,
                name=f"{dem.name}_{key}",
                source=dem.source,
                product=f"{dem.product} (dérivé : {key})",
                satellite=dem.satellite,
                variable="slope_pct" if key == "slope_percent" else "aspect_degrees",
                period_start=dem.period_start,
                period_end=dem.period_end,
                details={"derived_from": dem.id},
            )
        )
    return created


def compute_ndvi(
    session: Session, data_dir: Path, red_id: int, nir_id: int, name: str
) -> Dataset:
    red = get_dataset(session, red_id)
    nir = get_dataset(session, nir_id)
    output = data_dir / "processed" / "indices" / f"{name}.tif"
    calculate_ndvi(
        dataset_path(red, data_dir),
        dataset_path(nir, data_dir),
        output,
        source=red.source,
        product=red.product,
    )
    period_start = min(filter(None, [red.period_start, nir.period_start]), default=None)
    period_end = max(filter(None, [red.period_end, nir.period_end]), default=None)
    return register_raster(
        session,
        output,
        data_dir,
        name=name,
        source=red.source,
        product=f"{red.product} NDVI",
        satellite=red.satellite,
        variable="ndvi",
        period_start=period_start,
        period_end=period_end,
        details={"red_dataset_id": red.id, "nir_dataset_id": nir.id},
    )


def run_pipeline(config_dir: Path, data_dir: Path) -> dict[str, Any]:
    """Run the configured Kalehe pipeline (study_area.json + data_sources.json)."""
    return run_configured_pipeline(
        config_dir / "study_area.json",
        config_dir / "data_sources.json",
        data_dir / "processed" / "geospatial",
    )
