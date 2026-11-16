"""Query helpers shared by services and API routers."""

from __future__ import annotations

from typing import Any

from shapely.geometry import box
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .models import Dataset, Observation, Parcel


class NotFoundError(LookupError):
    """Raised when a requested entity does not exist."""


def get_parcel(session: Session, code: str) -> Parcel:
    parcel = session.scalar(select(Parcel).where(Parcel.code == code))
    if parcel is None:
        raise NotFoundError(f"Parcelle introuvable : {code}")
    return parcel


def get_dataset(session: Session, dataset_id: int) -> Dataset:
    dataset = session.get(Dataset, dataset_id)
    if dataset is None:
        raise NotFoundError(f"Jeu de données introuvable : {dataset_id}")
    return dataset


def list_parcels(
    session: Session,
    *,
    limit: int = 100,
    offset: int = 0,
    bbox: tuple[float, float, float, float] | None = None,
) -> tuple[list[Parcel], int]:
    """Return a page of parcels and the total count, optionally filtered by bbox.

    PostGIS evaluates the bbox with ``ST_Intersects`` (GIST index). Other dialects
    filter with Shapely after loading, which is only acceptable for small test sets.
    """
    statement = select(Parcel).order_by(Parcel.code)
    is_postgis = session.get_bind().dialect.name == "postgresql"
    if bbox is not None and is_postgis:
        envelope = func.ST_MakeEnvelope(*bbox, 4326)
        statement = statement.where(func.ST_Intersects(Parcel.geometry, envelope))
    if bbox is None or is_postgis:
        total = session.scalar(select(func.count()).select_from(statement.subquery()))
        parcels = list(session.scalars(statement.limit(limit).offset(offset)))
        return parcels, int(total or 0)
    window = box(*bbox)
    matches = [p for p in session.scalars(statement) if p.geometry.intersects(window)]
    return matches[offset : offset + limit], len(matches)


def latest_metrics(session: Session, parcel: Parcel) -> dict[str, Any]:
    """Return the most recent value per variable plus provenance information."""
    rows = session.scalars(
        select(Observation)
        .where(Observation.parcel_id == parcel.id)
        .order_by(
            Observation.variable,
            Observation.observed_end.desc().nulls_last(),
            Observation.created_at.desc(),
            Observation.id.desc(),
        )
    )
    metrics: dict[str, Any] = {}
    provenance: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.variable in metrics:
            continue
        metrics[row.variable] = row.value
        provenance[row.variable] = {
            "source": row.source,
            "method": row.method,
            "unit": row.unit,
            "observed_start": row.observed_start.isoformat() if row.observed_start else None,
            "observed_end": row.observed_end.isoformat() if row.observed_end else None,
            "is_demo": row.is_demo,
        }
    return {"metrics": metrics, "provenance": provenance}


def variable_series(session: Session, parcel: Parcel, variable: str) -> list[Observation]:
    return list(
        session.scalars(
            select(Observation)
            .where(Observation.parcel_id == parcel.id, Observation.variable == variable)
            .order_by(Observation.observed_end.asc().nulls_last(), Observation.id.asc())
        )
    )
