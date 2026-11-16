from __future__ import annotations

from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SAFE_NAME = r"^[A-Za-z0-9_-]{1,80}$"
CODE_PATTERN = r"^[A-Za-z0-9_.-]{1,80}$"


class _Out(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class ParcelOut(_Out):
    code: str
    name: str
    area_ha: float
    centroid_lat: float
    centroid_lon: float
    source: str
    is_demo: bool
    attributes: dict[str, Any]
    created_at: datetime | None = None
    updated_at: datetime | None = None


class ParcelPage(BaseModel):
    items: list[ParcelOut]
    total: int
    limit: int
    offset: int


class ObservationOut(_Out):
    id: int
    variable: str
    value: float
    unit: str | None
    observed_start: date | None
    observed_end: date | None
    source: str
    method: str
    dataset_id: int | None
    is_demo: bool
    quality: dict[str, Any]
    created_at: datetime | None = None


class DatasetOut(_Out):
    id: int
    name: str
    kind: str
    variable: str | None
    source: str
    product: str
    satellite: bool
    categorical: bool
    path: str
    crs: str | None
    resolution_m: float | None
    period_start: date | None
    period_end: date | None
    checksum_sha256: str | None
    details: dict[str, Any]
    created_at: datetime | None = None


class ImportReportOut(BaseModel):
    created: int
    updated: int
    skipped: int
    errors: list[str]
    warnings: list[str]


class ParcelMetrics(BaseModel):
    """Inputs of the decision engine; every field is optional (gaps are reported)."""

    model_config = ConfigDict(extra="forbid")

    slope_pct: float | None = Field(None, ge=0, le=1000)
    ndvi: float | None = Field(None, ge=-1, le=1)
    soil_moisture: float | None = Field(None, ge=0, le=1)
    root_zone_wetness: float | None = Field(None, ge=0, le=1)
    distance_to_forest_m: float | None = Field(None, ge=0)
    distance_to_village_km: float | None = Field(None, ge=0)
    distance_to_field_m: float | None = Field(None, ge=0)
    distance_to_protected_area_km: float | None = Field(None, ge=0)
    water_buffer_m: float | None = Field(None, ge=0)
    rainfall_mm: float | None = Field(None, ge=0, le=20000)
    temperature_c: float | None = Field(None, ge=-50, le=60)
    temperature_max_c: float | None = Field(None, ge=-50, le=70)
    land_surface_temperature_c: float | None = Field(None, ge=-50, le=90)
    canopy_height_m: float | None = Field(None, ge=0, le=120)
    elevation_m: float | None = Field(None, ge=-500, le=9000)
    cropland: bool | None = None
    degraded: bool | None = None


Matrix = list[list[float]]


class DecisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metrics: ParcelMetrics
    pairwise: dict[Literal["rna", "reboisement", "agroforesterie"], Matrix] | None = None


class PairwiseOverride(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pairwise: dict[Literal["rna", "reboisement", "agroforesterie"], Matrix] | None = None


class AHPRequest(BaseModel):
    matrix: Matrix = Field(..., min_length=1, max_length=10)


class ZonalRequest(BaseModel):
    dataset_ids: list[int] = Field(..., min_length=1, max_length=20)
    parcel_codes: list[str] | None = Field(None, max_length=1000)


class ProximityRequest(BaseModel):
    dataset_id: int
    variable: str = Field(..., pattern=r"^(distance_to_[a-z_]{1,40}_(m|km)|water_buffer_m)$")
    parcel_codes: list[str] | None = Field(None, max_length=1000)


class TerrainRequest(BaseModel):
    dataset_id: int


class NdviRequest(BaseModel):
    red_dataset_id: int
    nir_dataset_id: int
    name: str = Field(..., pattern=SAFE_NAME)


class RasterChangeRequest(BaseModel):
    before_dataset_id: int
    after_dataset_id: int
    threshold: float = Field(0.1, gt=0, le=1e6)
    name: str = Field(..., pattern=SAFE_NAME)
    variable: str = Field("ndvi", pattern=r"^[a-z_]{1,80}$")

    @model_validator(mode="after")
    def _distinct(self) -> "RasterChangeRequest":
        if self.before_dataset_id == self.after_dataset_id:
            raise ValueError("Les deux dates doivent provenir de jeux de données différents.")
        return self


class PowerParcelRequest(BaseModel):
    start: date
    end: date
    temporal: Literal["daily", "monthly", "climatology"] = "monthly"
    store: bool = True


class AppeearsTaskRequest(BaseModel):
    task_name: str = Field(..., pattern=SAFE_NAME)
    product_key: str | None = Field(None, pattern=r"^[a-z_]{1,40}$")
    layers: list[dict[str, str]] | None = Field(None, max_length=20)
    start: date
    end: date
    parcel_codes: list[str] | None = Field(None, max_length=1000)

    @model_validator(mode="after")
    def _layers(self) -> "AppeearsTaskRequest":
        if not self.product_key and not self.layers:
            raise ValueError("Fournir product_key ou layers.")
        return self


class AppeearsDownloadRequest(BaseModel):
    file_id: str = Field(..., pattern=r"^[0-9a-fA-F-]{8,64}$")
    file_name: str = Field(..., max_length=255)
    register_as: str | None = Field(None, pattern=SAFE_NAME)
    variable: str | None = Field(None, pattern=r"^[a-z_]{1,80}$")
    product: str | None = Field(None, max_length=200)
    period_start: date | None = None
    period_end: date | None = None
