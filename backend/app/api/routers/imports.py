from __future__ import annotations

from datetime import date

from fastapi import APIRouter, Depends, File, Form, UploadFile
from sqlalchemy.orm import Session

from ...core.config import Settings
from ...services.data_ingestion.importers import (
    TABLE_EXTENSIONS,
    VECTOR_EXTENSIONS,
    import_metrics_table,
    import_parcels,
    load_demo_data,
    read_vector,
    save_upload,
)
from ..deps import get_db, get_settings, require_api_key
from ..schemas import ImportReportOut

router = APIRouter(prefix="/imports", tags=["importation"], dependencies=[Depends(require_api_key)])


@router.post("/parcels", response_model=ImportReportOut, summary="Importer des parcelles (GeoJSON, GPKG, Shapefile .zip)")
def upload_parcels(
    file: UploadFile = File(...),
    source: str = Form(..., min_length=3, max_length=500),
    id_field: str = Form("id", max_length=80),
    name_field: str = Form("name", max_length=80),
    is_demo: bool = Form(False),
    session: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
):
    path = save_upload(file.file, file.filename, settings.upload_dir / "parcels",
                       VECTOR_EXTENSIONS, settings.max_upload_bytes)
    report = import_parcels(session, read_vector(path), source=source, id_field=id_field,
                            name_field=name_field, is_demo=is_demo)
    session.commit()
    return report.as_dict()


@router.post("/metrics", response_model=ImportReportOut, summary="Importer un tableau de mesures par parcelle (CSV)")
def upload_metrics(
    file: UploadFile = File(...),
    source: str = Form(..., min_length=3, max_length=500),
    observed_start: date | None = Form(None),
    observed_end: date | None = Form(None),
    id_column: str = Form("id", max_length=80),
    is_demo: bool = Form(False),
    session: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
):
    path = save_upload(file.file, file.filename, settings.upload_dir / "tables",
                       TABLE_EXTENSIONS, settings.max_upload_bytes)
    report = import_metrics_table(session, path, source=source, observed_start=observed_start,
                                  observed_end=observed_end, is_demo=is_demo, id_column=id_column)
    session.commit()
    return report.as_dict()


@router.post("/demo", summary="Charger les parcelles et mesures de démonstration du dépôt")
def import_demo(session: Session = Depends(get_db), settings: Settings = Depends(get_settings)) -> dict:
    report = load_demo_data(session, settings)
    session.commit()
    return {**report, "warning": "Données de démonstration : non validées, ne pas utiliser pour planter."}
