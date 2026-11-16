from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Path, Query, Response, status
from shapely.geometry import mapping
from sqlalchemy.orm import Session

from ...db.repositories import get_parcel, latest_metrics, list_parcels, variable_series
from ..deps import get_db, require_api_key
from ..schemas import CODE_PATTERN, ObservationOut, ParcelOut, ParcelPage

router = APIRouter(prefix="/parcels", tags=["parcelles"])

CodePath = Path(..., pattern=CODE_PATTERN)


def _parse_bbox(bbox: str | None) -> tuple[float, float, float, float] | None:
    if bbox is None:
        return None
    try:
        values = tuple(float(v) for v in bbox.split(","))
    except ValueError as exc:
        raise HTTPException(422, "bbox doit contenir 4 nombres : ouest,sud,est,nord.") from exc
    if len(values) != 4:
        raise HTTPException(422, "bbox doit contenir 4 nombres : ouest,sud,est,nord.")
    west, south, east, north = values
    if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
        raise HTTPException(422, "bbox hors limites WGS 84.")
    return values  # type: ignore[return-value]


@router.get("", response_model=ParcelPage, summary="Lister les parcelles")
def parcels(
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    bbox: str | None = Query(None, description="ouest,sud,est,nord (WGS 84)"),
    session: Session = Depends(get_db),
):
    items, total = list_parcels(session, limit=limit, offset=offset, bbox=_parse_bbox(bbox))
    return ParcelPage(items=items, total=total, limit=limit, offset=offset)


@router.get("/geojson", summary="Parcelles en GeoJSON (WGS 84)")
def parcels_geojson(
    limit: int = Query(500, ge=1, le=5000),
    bbox: str | None = Query(None),
    session: Session = Depends(get_db),
) -> dict:
    items, _ = list_parcels(session, limit=limit, bbox=_parse_bbox(bbox))
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "id": p.code,
                "geometry": mapping(p.geometry),
                "properties": {
                    "code": p.code,
                    "name": p.name,
                    "area_ha": p.area_ha,
                    "source": p.source,
                    "is_demo": p.is_demo,
                },
            }
            for p in items
        ],
    }


@router.get("/{code}", response_model=ParcelOut, summary="Détail d'une parcelle")
def parcel(code: str = CodePath, session: Session = Depends(get_db)):
    return get_parcel(session, code)


@router.get("/{code}/metrics", summary="Dernière valeur de chaque variable, avec provenance")
def parcel_metrics(code: str = CodePath, session: Session = Depends(get_db)) -> dict:
    return latest_metrics(session, get_parcel(session, code))


@router.get("/{code}/observations", response_model=list[ObservationOut], summary="Historique des observations")
def parcel_observations(
    code: str = CodePath,
    variable: str | None = Query(None, pattern=r"^[a-z_]{1,80}$"),
    session: Session = Depends(get_db),
):
    parcel = get_parcel(session, code)
    if variable:
        return variable_series(session, parcel, variable)
    return sorted(parcel.observations, key=lambda o: (o.variable, o.id))


@router.delete(
    "/{code}",
    status_code=status.HTTP_204_NO_CONTENT,
    dependencies=[Depends(require_api_key)],
    summary="Supprimer une parcelle et ses observations",
)
def delete_parcel(code: str = CodePath, session: Session = Depends(get_db)) -> Response:
    session.delete(get_parcel(session, code))
    session.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
