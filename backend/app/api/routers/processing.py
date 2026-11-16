from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ...core.config import Settings
from ...services import geoprocessing
from ..deps import get_db, get_settings, require_api_key
from ..schemas import DatasetOut, NdviRequest, ProximityRequest, TerrainRequest, ZonalRequest

router = APIRouter(prefix="/processing", tags=["traitement géospatial"],
                   dependencies=[Depends(require_api_key)])


@router.post("/pipeline", summary="Exécuter la chaîne Kalehe configurée (config/*.json)")
def run_pipeline(settings: Settings = Depends(get_settings)) -> dict:
    return geoprocessing.run_pipeline(settings.config_dir, settings.data_dir)


@router.post("/zonal-statistics", summary="Moyennes raster par parcelle → observations")
def zonal_statistics(body: ZonalRequest, session: Session = Depends(get_db),
                     settings: Settings = Depends(get_settings)) -> dict:
    result = geoprocessing.zonal_observations(session, settings.data_dir, body.dataset_ids, body.parcel_codes)
    session.commit()
    return result


@router.post("/proximity", summary="Distance parcelle → entité la plus proche (UTM 35S)")
def proximity(body: ProximityRequest, session: Session = Depends(get_db),
              settings: Settings = Depends(get_settings)) -> dict:
    result = geoprocessing.proximity_observations(
        session, settings.data_dir, body.dataset_id, body.variable, body.parcel_codes
    )
    session.commit()
    return result


@router.post("/terrain", response_model=list[DatasetOut], summary="Dériver pente (%) et exposition d'un MNT")
def terrain(body: TerrainRequest, session: Session = Depends(get_db),
            settings: Settings = Depends(get_settings)):
    created = geoprocessing.derive_terrain(session, settings.data_dir, body.dataset_id)
    session.commit()
    return created


@router.post("/ndvi", response_model=DatasetOut, summary="Calculer le NDVI à partir des bandes rouge/PIR")
def ndvi(body: NdviRequest, session: Session = Depends(get_db),
         settings: Settings = Depends(get_settings)):
    created = geoprocessing.compute_ndvi(session, settings.data_dir, body.red_dataset_id,
                                         body.nir_dataset_id, body.name)
    session.commit()
    return created
