from __future__ import annotations

from collections.abc import Callable
from datetime import date
from pathlib import Path as FsPath
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Path, Query, Request
from shapely.geometry import mapping
from sqlalchemy import select
from sqlalchemy.orm import Session

from ...core.config import Settings
from ...db.models import Parcel
from ...db.repositories import get_parcel
from ...services.data_ingestion.importers import add_observation, register_raster
from ...services.nasa import power
from ...services.nasa import appeears
from ...services.nasa import cmr
from ...services.nasa.catalog import NASA_PRODUCTS, get_product
from ...services.nasa.client import NasaHttpClient
from ..deps import get_db, get_settings, nasa_client_factory, require_api_key
from ..schemas import CODE_PATTERN, AppeearsDownloadRequest, AppeearsTaskRequest, PowerParcelRequest

router = APIRouter(prefix="/nasa", tags=["API NASA"])
ClientFactory = Callable[[str], NasaHttpClient]
TaskIdPath = Path(..., pattern=r"^[0-9a-fA-F-]{8,64}$")


@router.get("/products", summary="Produits NASA mobilisés par le modèle")
def products() -> list[dict]:
    return [p.as_dict() for p in NASA_PRODUCTS.values()]


# ----------------------------------------------------------------- POWER
def _parameters(raw: str | None) -> tuple[str, ...]:
    if not raw:
        return power.DEFAULT_PARAMETERS
    return tuple(p.strip().upper() for p in raw.split(",") if p.strip())


@router.get("/power/point", summary="Climatologie agricole NASA POWER pour un point")
def power_point(
    latitude: float = Query(..., ge=-90, le=90),
    longitude: float = Query(..., ge=-180, le=180),
    start: date = Query(...),
    end: date = Query(...),
    temporal: Literal["daily", "monthly", "climatology"] = "monthly",
    parameters: str | None = Query(None, max_length=400, description="Liste séparée par des virgules"),
    clients: ClientFactory = Depends(nasa_client_factory),
) -> dict:
    request = power.PowerRequest(latitude, longitude, start, end, temporal, _parameters(parameters))
    with clients(power.SERVICE) as client:
        return power.fetch_point(client, request)


@router.post(
    "/power/parcels/{code}",
    dependencies=[Depends(require_api_key)],
    summary="Récupérer POWER au centroïde d'une parcelle et stocker les indicateurs",
)
def power_for_parcel(
    body: PowerParcelRequest,
    code: str = Path(..., pattern=CODE_PATTERN),
    session: Session = Depends(get_db),
    clients: ClientFactory = Depends(nasa_client_factory),
) -> dict:
    parcel = get_parcel(session, code)
    request = power.PowerRequest(parcel.centroid_lat, parcel.centroid_lon, body.start, body.end, body.temporal)
    with clients(power.SERVICE) as client:
        result = power.fetch_point(client, request)
    stored = {}
    if body.store:
        for variable, (value, unit) in power.to_parcel_metrics(result).items():
            add_observation(
                session, parcel, variable, value, unit=unit,
                source="NASA POWER (power.larc.nasa.gov)", method=f"nasa_power_{body.temporal}",
                observed_start=body.start, observed_end=body.end,
                quality={"resolution": "~0,5° x 0,625°", "point": [parcel.centroid_lon, parcel.centroid_lat]},
                is_demo=parcel.is_demo,
            )
            stored[variable] = value
        session.commit()
    return {"parcel": code, "stored": stored, "power": result}


# ----------------------------------------------------------------- CMIP6
@router.get("/cmip6/synthese", summary="Scénarios climatiques CMIP6 (NASA NEX-GDDP) résumés sur Kalehe")
def cmip6_synthese(
    scenario: Literal["ssp126", "ssp245", "ssp370", "ssp585"] | None = None,
    periode: Literal["2021-2040", "2041-2060", "2081-2100"] | None = None,
    settings: Settings = Depends(get_settings),
) -> dict:
    """Anomalies multi-modèles (médiane, 10e-90e centiles) par rapport à 1995-2014, produites par
    ``scripts/telecharger_cmip6.py`` ; avec ``scenario`` et ``periode``, ajoute les implications."""
    import json

    import pandas as pd

    from ...analysis import climat

    dossier = settings.data_dir / "processed" / "climat"
    chemin, meta = dossier / "cmip6_synthese.csv", dossier / "cmip6_meta.json"
    if not chemin.is_file() or not meta.is_file():
        raise HTTPException(404, "Synthèse CMIP6 absente : exécuter scripts/telecharger_cmip6.py.")
    synthese = pd.read_csv(chemin)
    selection = synthese
    if scenario:
        selection = selection[selection["scenario"] == scenario]
    if periode:
        selection = selection[selection["periode"] == periode]
    reponse: dict = {
        "meta": json.loads(meta.read_text(encoding="utf-8")),
        "synthese": selection.astype(object).where(pd.notna(selection), None).to_dict("records"),
    }
    reponse["meta"].pop("grille", None)
    if scenario and periode:
        reponse["implications"] = climat.implications(synthese, scenario, periode)
    return reponse


# ------------------------------------------------------------------- CMR
def _bbox_for(session: Session, bbox: str | None, parcel_code: str | None) -> tuple[float, float, float, float]:
    if bbox:
        try:
            values = tuple(float(v) for v in bbox.split(","))
        except ValueError as exc:
            raise HTTPException(422, "bbox invalide.") from exc
        if len(values) != 4:
            raise HTTPException(422, "bbox doit contenir 4 nombres.")
        return values  # type: ignore[return-value]
    if parcel_code:
        return tuple(get_parcel(session, parcel_code).geometry.bounds)  # type: ignore[return-value]
    raise HTTPException(422, "Fournir bbox ou parcel_code.")


@router.get("/cmr/granules", summary="Rechercher des granules NASA (CMR) sur une zone")
def cmr_granules(
    product: str = Query(..., pattern=r"^[a-z_]{1,40}$"),
    start: date = Query(...),
    end: date = Query(...),
    bbox: str | None = Query(None, description="ouest,sud,est,nord"),
    parcel_code: str | None = Query(None, pattern=CODE_PATTERN),
    page_size: int = Query(50, ge=1, le=cmr.MAX_PAGE_SIZE),
    max_cloud_cover: float | None = Query(None, ge=0, le=100),
    session: Session = Depends(get_db),
    clients: ClientFactory = Depends(nasa_client_factory),
) -> dict:
    try:
        selected = get_product(product)
    except KeyError as exc:
        raise HTTPException(404, str(exc.args[0])) from exc
    area = _bbox_for(session, bbox, parcel_code)
    with clients(cmr.SERVICE) as client:
        return cmr.search_granules(client, selected, area, start, end, page_size=page_size,
                                   max_cloud_cover=max_cloud_cover)


# -------------------------------------------------------------- AppEEARS
def _appeears_client(request: Request, clients: ClientFactory, settings: Settings) -> NasaHttpClient:
    client = clients(appeears.SERVICE)
    token = getattr(request.app.state, "appeears_token", None)
    if token:
        client.bearer_token = token
    else:
        password = settings.earthdata_password.get_secret_value() if settings.earthdata_password else None
        try:
            request.app.state.appeears_token = appeears.login(client, settings.earthdata_username, password)
        except Exception:
            client.close()
            raise
    return client


@router.post("/appeears/tasks", dependencies=[Depends(require_api_key)],
             summary="Soumettre une extraction AppEEARS sur les parcelles")
def appeears_submit(
    body: AppeearsTaskRequest,
    request: Request,
    session: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
    clients: ClientFactory = Depends(nasa_client_factory),
) -> dict:
    if body.layers:
        layers = body.layers
    else:
        try:
            product = get_product(body.product_key or "")
        except KeyError as exc:
            raise HTTPException(404, str(exc.args[0])) from exc
        if not product.appeears_product:
            raise HTTPException(422, f"{product.short_name} n'est pas disponible via AppEEARS dans ce catalogue.")
        layers = [{"product": product.appeears_product, "layer": layer} for layer in product.appeears_layers]
    statement = select(Parcel).order_by(Parcel.code)
    if body.parcel_codes:
        statement = statement.where(Parcel.code.in_(body.parcel_codes))
    parcels = list(session.scalars(statement.limit(1000)))
    if not parcels:
        raise HTTPException(422, "Aucune parcelle pour définir la zone d'extraction.")
    geo = {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "properties": {"code": p.code}, "geometry": mapping(p.geometry)}
            for p in parcels
        ],
    }
    task = appeears.build_area_task(body.task_name, geo, layers, body.start, body.end)
    client = _appeears_client(request, clients, settings)
    with client:
        return appeears.submit_task(client, task)


@router.get("/appeears/tasks/{task_id}", dependencies=[Depends(require_api_key)])
def appeears_status(
    request: Request,
    task_id: str = TaskIdPath,
    settings: Settings = Depends(get_settings),
    clients: ClientFactory = Depends(nasa_client_factory),
) -> dict:
    with _appeears_client(request, clients, settings) as client:
        return appeears.task_status(client, task_id)


@router.get("/appeears/tasks/{task_id}/bundle", dependencies=[Depends(require_api_key)])
def appeears_bundle(
    request: Request,
    task_id: str = TaskIdPath,
    settings: Settings = Depends(get_settings),
    clients: ClientFactory = Depends(nasa_client_factory),
) -> dict:
    with _appeears_client(request, clients, settings) as client:
        return appeears.bundle(client, task_id)


@router.post("/appeears/tasks/{task_id}/download", dependencies=[Depends(require_api_key)],
             summary="Télécharger un fichier AppEEARS et l'enregistrer comme jeu de données")
def appeears_download(
    body: AppeearsDownloadRequest,
    request: Request,
    task_id: str = TaskIdPath,
    session: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
    clients: ClientFactory = Depends(nasa_client_factory),
) -> dict:
    with _appeears_client(request, clients, settings) as client:
        path = appeears.download_bundle_file(
            client, task_id, body.file_id, body.file_name,
            settings.nasa_raw_dir / "appeears", int(settings.nasa_max_download_mb * 1024 * 1024),
        )
    response: dict = {"path": FsPath(path).relative_to(settings.data_dir).as_posix()}
    if body.register_as and path.suffix.lower() in {".tif", ".tiff"}:
        dataset = register_raster(
            session, path, settings.data_dir, name=body.register_as,
            source="NASA AppEEARS (LP DAAC)", product=body.product or body.file_name,
            satellite=True, variable=body.variable,
            period_start=body.period_start, period_end=body.period_end,
            details={"appeears_task_id": task_id, "file_id": body.file_id},
        )
        session.commit()
        response["dataset_id"] = dataset.id
    return response
