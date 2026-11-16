from __future__ import annotations

from collections.abc import Callable
from datetime import date
from pathlib import Path as FsPath

from fastapi import APIRouter, Depends, HTTPException, Path, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from ...analysis.change_detection import raster_change
from ...analysis.timeseries import analyze_series, climate_anomaly
from ...core.config import Settings
from ...db.models import ChangeAnalysis
from ...db.repositories import get_dataset, get_parcel, variable_series
from ...processing.features import extract_zonal_statistics
from ...services.data_ingestion.importers import dataset_path, register_raster
from ...services.geoprocessing import parcels_frame
from ...services.nasa import power
from ...services.nasa.client import NasaHttpClient
from ..deps import get_db, get_settings, nasa_client_factory, require_api_key
from ..schemas import CODE_PATTERN, RasterChangeRequest

router = APIRouter(prefix="/analysis", tags=["analyse des changements"])


@router.get("", summary="Analyses de changement enregistrées")
def analyses(limit: int = Query(50, ge=1, le=500), session: Session = Depends(get_db)) -> list[dict]:
    rows = session.scalars(select(ChangeAnalysis).order_by(ChangeAnalysis.id.desc()).limit(limit))
    return [
        {"id": r.id, "kind": r.kind, "variable": r.variable, "before_dataset_id": r.before_dataset_id,
         "after_dataset_id": r.after_dataset_id, "created_at": r.created_at, "result": r.result}
        for r in rows
    ]


@router.post("/raster-change", dependencies=[Depends(require_api_key)],
             summary="Différence entre deux dates (même grille) + classes perte/stable/gain")
def raster_change_endpoint(
    body: RasterChangeRequest,
    session: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> dict:
    before = get_dataset(session, body.before_dataset_id)
    after = get_dataset(session, body.after_dataset_id)
    if before.kind != "raster" or after.kind != "raster":
        raise HTTPException(422, "Les deux jeux de données doivent être des rasters.")
    if before.categorical or after.categorical:
        raise HTTPException(422, "L'analyse de différence exige des rasters continus.")
    if before.period_end and after.period_start and before.period_end > after.period_start:
        raise HTTPException(422, "La période « avant » doit précéder la période « après ».")

    output_dir = settings.processed_dir / "change" / body.name
    result = raster_change(dataset_path(before, settings.data_dir), dataset_path(after, settings.data_dir),
                           output_dir, threshold=body.threshold, name=body.name)
    common = dict(source=before.source, satellite=before.satellite,
                  period_start=before.period_start, period_end=after.period_end,
                  details={"before_dataset_id": before.id, "after_dataset_id": after.id})
    difference = register_raster(session, FsPath(result["difference_path"]), settings.data_dir,
                                 name=f"{body.name}_difference", product=f"Δ {before.product}",
                                 variable=f"{body.variable}_change", **common)
    classes = register_raster(session, FsPath(result["classes_path"]), settings.data_dir,
                              name=f"{body.name}_classes", product=f"Classes Δ {before.product}",
                              variable=f"{body.variable}_change_class", categorical=True, **common)

    per_parcel = []
    try:
        zones = parcels_frame(session)
    except Exception:
        zones = None
    if zones is not None:
        for _, zone in zones.iterrows():
            try:
                stats = extract_zonal_statistics({"diff": result["difference_path"]}, zones.loc[[zone.name]],
                                                 zone_id_field="code")
            except Exception:
                continue  # parcel outside the analysed rasters
            row = stats.iloc[0]
            per_parcel.append({"parcel": zone.code, "mean_difference": float(row["mean"]),
                               "valid_pixel_count": int(row["valid_pixel_count"])})

    stored_result = {**result, "per_parcel": per_parcel,
                     "difference_path": difference.path, "classes_path": classes.path}
    record = ChangeAnalysis(kind="raster_difference", variable=body.variable,
                            before_dataset_id=before.id, after_dataset_id=after.id, result=stored_result)
    session.add(record)
    session.commit()
    return {"id": record.id, "difference_dataset_id": difference.id,
            "classes_dataset_id": classes.id, **stored_result}


@router.get("/parcels/{code}/trend", summary="Tendance temporelle d'une variable (Mann-Kendall, Sen)")
def parcel_trend(
    code: str = Path(..., pattern=CODE_PATTERN),
    variable: str = Query("ndvi", pattern=r"^[a-z_]{1,80}$"),
    alpha: float = Query(0.05, gt=0, lt=0.5),
    session: Session = Depends(get_db),
) -> dict:
    parcel = get_parcel(session, code)
    series = variable_series(session, parcel, variable)
    points = [(o.observed_end or o.observed_start, o.value) for o in series]
    undated = sum(1 for d, _ in points if d is None)
    result = analyze_series([(d, v) for d, v in points if d is not None], alpha=alpha)
    return {"parcel": code, "variable": variable, "undated_observations_ignored": undated,
            "based_on_demo_data": any(o.is_demo for o in series), **result}


@router.get("/climate-anomaly", summary="Anomalie climatique NASA POWER (période cible vs référence)")
def climate_anomaly_endpoint(
    latitude: float = Query(..., ge=-90, le=90),
    longitude: float = Query(..., ge=-180, le=180),
    parameter: str = Query("PRECTOTCORR", pattern=r"^[A-Z0-9_]{2,40}$"),
    baseline_start: int = Query(1991, ge=power.FIRST_YEAR),
    baseline_end: int = Query(2020, ge=power.FIRST_YEAR),
    target_start: int = Query(2021, ge=power.FIRST_YEAR),
    target_end: int = Query(..., ge=power.FIRST_YEAR),
    clients: Callable[[str], NasaHttpClient] = Depends(nasa_client_factory),
) -> dict:
    if not (baseline_start <= baseline_end and target_start <= target_end):
        raise HTTPException(422, "Périodes invalides.")
    first, last = min(baseline_start, target_start), max(baseline_end, target_end)
    request = power.PowerRequest(latitude, longitude, date(first, 1, 1),
                                 min(date(last, 12, 31), date.today()), "monthly", (parameter,))
    with clients(power.SERVICE) as client:
        result = power.fetch_point(client, request)
    records = result["series"].get(parameter, [])
    anomaly = climate_anomaly(records, (baseline_start, baseline_end), (target_start, target_end))
    return {"parameter": parameter, "unit": result["units"].get(parameter),
            "baseline": [baseline_start, baseline_end], "target": [target_start, target_end],
            **anomaly, "resolution_note": result["resolution_note"]}
