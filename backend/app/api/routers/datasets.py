from __future__ import annotations

from datetime import date

from fastapi import APIRouter, Depends, File, Form, Query, Response, UploadFile, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from ...core.config import Settings
from ...core.security import UnsafePathError, ensure_within
from ...db.models import Dataset
from ...db.repositories import get_dataset
from ...services.data_ingestion.importers import (
    RASTER_EXTENSIONS,
    VECTOR_EXTENSIONS,
    dataset_path,
    register_raster,
    register_vector,
    save_upload,
)
from ..deps import get_db, get_settings, require_api_key
from ..schemas import SAFE_NAME, DatasetOut

router = APIRouter(prefix="/datasets", tags=["jeux de données"])


@router.get("", response_model=list[DatasetOut], summary="Lister les jeux de données enregistrés")
def datasets(
    kind: str | None = Query(None, pattern="^(raster|vector)$"),
    session: Session = Depends(get_db),
):
    statement = select(Dataset).order_by(Dataset.id)
    if kind:
        statement = statement.where(Dataset.kind == kind)
    return list(session.scalars(statement))


@router.get("/{dataset_id}", response_model=DatasetOut)
def dataset(dataset_id: int, session: Session = Depends(get_db)):
    return get_dataset(session, dataset_id)


@router.post(
    "/raster",
    response_model=DatasetOut,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_api_key)],
    summary="Téléverser un GeoTIFF mono-bande avec sa provenance",
)
def upload_raster(
    file: UploadFile = File(...),
    name: str = Form(..., pattern=SAFE_NAME),
    source: str = Form(..., min_length=3, max_length=500),
    product: str = Form(..., min_length=2, max_length=200),
    satellite: bool = Form(...),
    variable: str | None = Form(None, pattern=r"^[a-z_]{1,80}$"),
    categorical: bool = Form(False),
    period_start: date | None = Form(None),
    period_end: date | None = Form(None),
    session: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
):
    path = save_upload(file.file, file.filename, settings.upload_dir / "rasters",
                       RASTER_EXTENSIONS, settings.max_upload_bytes)
    try:
        created = register_raster(
            session, path, settings.data_dir, name=name, source=source, product=product,
            satellite=satellite, variable=variable, categorical=categorical,
            period_start=period_start, period_end=period_end,
        )
    except Exception:
        path.unlink(missing_ok=True)
        raise
    session.commit()
    return created


@router.post(
    "/vector",
    response_model=DatasetOut,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_api_key)],
    summary="Téléverser une couche de référence (forêts, villages, sources…)",
)
def upload_vector(
    file: UploadFile = File(...),
    name: str = Form(..., pattern=SAFE_NAME),
    source: str = Form(..., min_length=3, max_length=500),
    product: str = Form(..., min_length=2, max_length=200),
    satellite: bool = Form(False),
    variable: str | None = Form(None, pattern=r"^[a-z_]{1,80}$"),
    session: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
):
    path = save_upload(file.file, file.filename, settings.upload_dir / "vectors",
                       VECTOR_EXTENSIONS, settings.max_upload_bytes)
    try:
        created = register_vector(session, path, settings.data_dir, name=name, source=source,
                                  product=product, satellite=satellite, variable=variable)
    except Exception:
        path.unlink(missing_ok=True)
        raise
    session.commit()
    return created


@router.delete(
    "/{dataset_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    dependencies=[Depends(require_api_key)],
    summary="Désenregistrer un jeu de données (le fichier n'est supprimé que s'il a été téléversé)",
)
def delete_dataset(
    dataset_id: int,
    session: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> Response:
    record = get_dataset(session, dataset_id)
    path = dataset_path(record, settings.data_dir)
    session.delete(record)
    session.commit()
    try:
        ensure_within(path, settings.upload_dir)
        path.unlink(missing_ok=True)
    except UnsafePathError:
        pass  # files outside uploads/ (e.g. curated sources) are never deleted by the API
    return Response(status_code=status.HTTP_204_NO_CONTENT)
