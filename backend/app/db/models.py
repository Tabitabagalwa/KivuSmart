"""ORM models. Geometries are stored in WGS 84 (EPSG:4326); metric computations
are done in UTM 35S (EPSG:32735) by the services."""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base, JSONType
from .types import GeometryType


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class StudyArea(TimestampMixin, Base):
    __tablename__ = "study_areas"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(120), unique=True)
    source: Mapped[str] = mapped_column(String(500))
    geometry: Mapped[Any] = mapped_column(GeometryType("MULTIPOLYGON"))
    is_authoritative: Mapped[bool] = mapped_column(Boolean, default=False)


class Parcel(TimestampMixin, Base):
    __tablename__ = "parcels"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    code: Mapped[str] = mapped_column(String(80), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(200))
    study_area_id: Mapped[int | None] = mapped_column(
        ForeignKey("study_areas.id", ondelete="SET NULL"), nullable=True
    )
    geometry: Mapped[Any] = mapped_column(GeometryType("MULTIPOLYGON"))
    area_ha: Mapped[float] = mapped_column(Float)
    centroid_lat: Mapped[float] = mapped_column(Float)
    centroid_lon: Mapped[float] = mapped_column(Float)
    source: Mapped[str] = mapped_column(String(500))
    is_demo: Mapped[bool] = mapped_column(Boolean, default=False)
    attributes: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )

    observations: Mapped[list["Observation"]] = relationship(
        back_populates="parcel", cascade="all, delete-orphan", passive_deletes=True
    )


class Dataset(TimestampMixin, Base):
    """A registered source file (raster or vector) with explicit provenance."""

    __tablename__ = "datasets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(120), unique=True)
    kind: Mapped[str] = mapped_column(String(20))  # raster | vector
    variable: Mapped[str | None] = mapped_column(String(80), nullable=True)
    source: Mapped[str] = mapped_column(String(500))
    product: Mapped[str] = mapped_column(String(200))
    satellite: Mapped[bool] = mapped_column(Boolean)
    categorical: Mapped[bool] = mapped_column(Boolean, default=False)
    path: Mapped[str] = mapped_column(Text)  # relative to the data directory
    crs: Mapped[str | None] = mapped_column(String(200), nullable=True)
    resolution_m: Mapped[float | None] = mapped_column(Float, nullable=True)
    period_start: Mapped[date | None] = mapped_column(Date, nullable=True)
    period_end: Mapped[date | None] = mapped_column(Date, nullable=True)
    checksum_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    details: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict)


class Observation(TimestampMixin, Base):
    """One measured or derived value of a variable for a parcel."""

    __tablename__ = "observations"
    __table_args__ = (
        Index("ix_observations_parcel_variable", "parcel_id", "variable"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    parcel_id: Mapped[int] = mapped_column(ForeignKey("parcels.id", ondelete="CASCADE"))
    variable: Mapped[str] = mapped_column(String(80))
    value: Mapped[float] = mapped_column(Float)
    unit: Mapped[str | None] = mapped_column(String(40), nullable=True)
    observed_start: Mapped[date | None] = mapped_column(Date, nullable=True)
    observed_end: Mapped[date | None] = mapped_column(Date, nullable=True)
    source: Mapped[str] = mapped_column(String(500))
    method: Mapped[str] = mapped_column(String(60))
    dataset_id: Mapped[int | None] = mapped_column(
        ForeignKey("datasets.id", ondelete="SET NULL"), nullable=True
    )
    is_demo: Mapped[bool] = mapped_column(Boolean, default=False)
    quality: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict)

    parcel: Mapped[Parcel] = relationship(back_populates="observations")


class NasaRequestLog(TimestampMixin, Base):
    __tablename__ = "nasa_request_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    service: Mapped[str] = mapped_column(String(40))
    url: Mapped[str] = mapped_column(Text)
    status_code: Mapped[int | None] = mapped_column(Integer, nullable=True)
    ok: Mapped[bool] = mapped_column(Boolean)
    cached: Mapped[bool] = mapped_column(Boolean, default=False)
    duration_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)


class ChangeAnalysis(TimestampMixin, Base):
    __tablename__ = "change_analyses"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    kind: Mapped[str] = mapped_column(String(40))  # raster_difference | metric_trend
    variable: Mapped[str] = mapped_column(String(80))
    parcel_id: Mapped[int | None] = mapped_column(
        ForeignKey("parcels.id", ondelete="CASCADE"), nullable=True
    )
    before_dataset_id: Mapped[int | None] = mapped_column(
        ForeignKey("datasets.id", ondelete="SET NULL"), nullable=True
    )
    after_dataset_id: Mapped[int | None] = mapped_column(
        ForeignKey("datasets.id", ondelete="SET NULL"), nullable=True
    )
    result: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict)


class Recommendation(TimestampMixin, Base):
    __tablename__ = "recommendations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    parcel_id: Mapped[int] = mapped_column(ForeignKey("parcels.id", ondelete="CASCADE"))
    strategy_id: Mapped[str] = mapped_column(String(40))
    score: Mapped[float | None] = mapped_column(Float, nullable=True)
    confidence: Mapped[float] = mapped_column(Float)
    engine_version: Mapped[str] = mapped_column(String(20))
    based_on_demo_data: Mapped[bool] = mapped_column(Boolean, default=False)
    result: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict)
