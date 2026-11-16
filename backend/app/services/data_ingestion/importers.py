"""Validated import of parcels, parcel metrics and raster/vector datasets.

Rules carried over from the processing package: CRS must be declared (never
guessed), satellite provenance must be NASA, Sentinel inputs are refused, and
demonstration data stay flagged as such.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import zipfile
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, BinaryIO

import geopandas as gpd
import pandas as pd
from shapely import make_valid
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry
from sqlalchemy import select
from sqlalchemy.orm import Session

from ...core.security import ensure_within, safe_filename
from ...db.models import Dataset, Observation, Parcel
from ...processing.errors import GeospatialProcessingError
from ...processing.raster.harmonize import RasterLayer, _validate_layer_provenance
from ...processing.raster.metadata import inspect_raster
from ...processing.vector.boundaries import _polygon_parts

METRIC_CRS = "EPSG:32735"  # UTM 35S, metric CRS used across the project
STORAGE_CRS = "EPSG:4326"

VECTOR_EXTENSIONS = {".geojson", ".json", ".gpkg", ".zip"}
RASTER_EXTENSIONS = {".tif", ".tiff"}
TABLE_EXTENSIONS = {".csv"}
MAX_ZIP_UNCOMPRESSED_BYTES = 500 * 1024 * 1024
DEMO_SOURCE = "Données de démonstration du dépôt (non validées sur le terrain)"


@dataclass(frozen=True)
class VariableSpec:
    unit: str
    minimum: float | None = None
    maximum: float | None = None
    boolean: bool = False


# Variables understood by the decision engine, with physical validity ranges.
VARIABLE_CATALOG: dict[str, VariableSpec] = {
    "slope_pct": VariableSpec("%", 0, 1000),
    "ndvi": VariableSpec("sans unité", -1, 1),
    "soil_moisture": VariableSpec("m³/m³", 0, 1),
    "root_zone_wetness": VariableSpec("0-1", 0, 1),
    "distance_to_forest_m": VariableSpec("m", 0, None),
    "distance_to_village_km": VariableSpec("km", 0, None),
    "distance_to_field_m": VariableSpec("m", 0, None),
    "distance_to_protected_area_km": VariableSpec("km", 0, None),
    "water_buffer_m": VariableSpec("m", 0, None),
    "rainfall_mm": VariableSpec("mm/an", 0, 20000),
    "temperature_c": VariableSpec("°C", -50, 60),
    "temperature_max_c": VariableSpec("°C", -50, 70),
    "land_surface_temperature_c": VariableSpec("°C", -50, 90),
    "canopy_height_m": VariableSpec("m", 0, 120),
    "elevation_m": VariableSpec("m", -500, 9000),
    "cropland": VariableSpec("booléen", boolean=True),
    "degraded": VariableSpec("booléen", boolean=True),
}


class ImportValidationError(ValueError):
    """Raised when an uploaded file cannot be imported safely."""


@dataclass
class ImportReport:
    created: int = 0
    updated: int = 0
    skipped: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------- uploads
def save_upload(
    stream: BinaryIO,
    original_name: str | None,
    destination_dir: Path,
    allowed_extensions: set[str],
    max_bytes: int,
) -> Path:
    """Persist an uploaded stream under a sanitized, unique name."""
    suffix = Path(original_name or "").suffix.lower()
    if suffix not in allowed_extensions:
        raise ImportValidationError(
            f"Extension {suffix or '(aucune)'} refusée. Autorisées : {sorted(allowed_extensions)}"
        )
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = ensure_within(destination_dir / safe_filename(original_name), destination_dir)
    written = 0
    try:
        with destination.open("wb") as handle:
            while chunk := stream.read(1024 * 1024):
                written += len(chunk)
                if written > max_bytes:
                    raise ImportValidationError("Fichier trop volumineux.")
                handle.write(chunk)
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    if written == 0:
        destination.unlink(missing_ok=True)
        raise ImportValidationError("Fichier vide.")
    if suffix == ".zip":
        _check_zip(destination)
    return destination


def _check_zip(path: Path) -> None:
    try:
        with zipfile.ZipFile(path) as archive:
            total = 0
            for info in archive.infolist():
                name = info.filename
                if name.startswith(("/", "\\")) or ".." in Path(name).parts:
                    raise ImportValidationError("Archive ZIP contenant des chemins dangereux.")
                total += info.file_size
            if total > MAX_ZIP_UNCOMPRESSED_BYTES:
                raise ImportValidationError("Archive ZIP trop volumineuse une fois décompressée.")
            if not any(n.lower().endswith(".shp") for n in archive.namelist()):
                raise ImportValidationError("L'archive ZIP doit contenir un Shapefile (.shp).")
    except zipfile.BadZipFile as exc:
        path.unlink(missing_ok=True)
        raise ImportValidationError("Archive ZIP invalide.") from exc


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------- vector
def read_vector(path: Path) -> gpd.GeoDataFrame:
    """Read a vector file; its CRS must be declared (GeoJSON is WGS 84 by spec)."""
    source = f"zip://{path}" if path.suffix.lower() == ".zip" else str(path)
    try:
        frame = gpd.read_file(source)
    except Exception as exc:
        raise ImportValidationError(f"Lecture du fichier vectoriel impossible : {exc}") from exc
    if frame.empty:
        raise ImportValidationError("Le fichier vectoriel ne contient aucune entité.")
    if frame.crs is None:
        raise ImportValidationError(
            "Le fichier vectoriel n'a pas de SCR déclaré ; il n'est jamais deviné."
        )
    return frame


def _as_multipolygon(geometry: BaseGeometry | None) -> MultiPolygon | None:
    if geometry is None or geometry.is_empty:
        return None
    parts = [p for p in _polygon_parts(make_valid(geometry)) if not p.is_empty and p.area > 0]
    if not parts:
        return None
    return MultiPolygon(parts)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if hasattr(value, "item"):  # numpy scalar
        return _json_safe(value.item())
    if isinstance(value, (date, pd.Timestamp)):
        return value.isoformat()
    return str(value)


def import_parcels(
    session: Session,
    frame: gpd.GeoDataFrame,
    *,
    source: str,
    id_field: str = "id",
    name_field: str = "name",
    is_demo: bool = False,
    study_area_id: int | None = None,
) -> ImportReport:
    """Upsert polygon parcels keyed by ``id_field``. Geometries are repaired and
    stored in WGS 84; area and centroid are computed in UTM 35S."""
    if not source or not source.strip():
        raise ImportValidationError("La provenance (source) des parcelles est obligatoire.")
    if id_field not in frame.columns:
        raise ImportValidationError(f"Champ identifiant absent : {id_field}")
    report = ImportReport()
    storage = frame.to_crs(STORAGE_CRS)
    metric = frame.to_crs(METRIC_CRS)
    seen: set[str] = set()
    for index, row in storage.iterrows():
        code = str(row[id_field]).strip() if row[id_field] is not None else ""
        if not code or code.lower() == "nan" or len(code) > 80:
            report.skipped += 1
            report.errors.append(f"Entité {index} : identifiant manquant ou trop long.")
            continue
        if code in seen:
            report.skipped += 1
            report.errors.append(f"Identifiant dupliqué dans le fichier : {code}")
            continue
        seen.add(code)
        geometry = _as_multipolygon(row.geometry)
        metric_geometry = _as_multipolygon(metric.geometry.loc[index])
        if geometry is None or metric_geometry is None:
            report.skipped += 1
            report.errors.append(f"{code} : géométrie non polygonale ou vide.")
            continue
        centroid = gpd.GeoSeries([metric_geometry.centroid], crs=METRIC_CRS).to_crs(STORAGE_CRS).iloc[0]
        name_value = row[name_field] if name_field in storage.columns else None
        name = str(name_value).strip() if name_value not in (None, "") else code
        attributes = {
            str(k): _json_safe(v)
            for k, v in row.items()
            if k not in {"geometry", id_field, name_field}
        }
        values = {
            "name": name[:200],
            "geometry": geometry,
            "area_ha": round(metric_geometry.area / 10_000, 4),
            "centroid_lat": round(centroid.y, 6),
            "centroid_lon": round(centroid.x, 6),
            "source": source.strip()[:500],
            "is_demo": is_demo,
            "attributes": attributes,
            "study_area_id": study_area_id,
        }
        parcel = session.scalar(select(Parcel).where(Parcel.code == code))
        if parcel is None:
            session.add(Parcel(code=code, **values))
            report.created += 1
        else:
            for key, value in values.items():
                setattr(parcel, key, value)
            report.updated += 1
    session.flush()
    return report


# ---------------------------------------------------------------------- tables
def _parse_bool(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    text = str(value).strip().casefold()
    if text in {"true", "1", "oui", "yes", "vrai"}:
        return 1.0
    if text in {"false", "0", "non", "no", "faux"}:
        return 0.0
    return None


def import_metrics_table(
    session: Session,
    content: bytes | str | Path,
    *,
    source: str,
    observed_start: date | None = None,
    observed_end: date | None = None,
    is_demo: bool = False,
    id_column: str = "id",
) -> ImportReport:
    """Import a wide CSV (one row per parcel, one column per variable)."""
    if not source or not source.strip():
        raise ImportValidationError("La provenance (source) des mesures est obligatoire.")
    if observed_start and observed_end and observed_start > observed_end:
        raise ImportValidationError("observed_start est postérieure à observed_end.")
    try:
        if isinstance(content, Path):
            table = pd.read_csv(content, dtype=str, keep_default_na=False)
        else:
            raw = content.encode("utf-8") if isinstance(content, str) else content
            table = pd.read_csv(io.BytesIO(raw), dtype=str, keep_default_na=False)
    except Exception as exc:
        raise ImportValidationError(f"CSV illisible : {exc}") from exc
    if id_column not in table.columns:
        raise ImportValidationError(f"Colonne identifiant absente : {id_column}")

    report = ImportReport()
    known = [c for c in table.columns if c in VARIABLE_CATALOG]
    ignored = [c for c in table.columns if c not in VARIABLE_CATALOG and c not in {id_column, "name"}]
    if ignored:
        report.warnings.append(f"Colonnes ignorées (variables inconnues) : {ignored}")
    if not known:
        raise ImportValidationError("Aucune variable reconnue dans le CSV.")

    parcels = {p.code: p for p in session.scalars(select(Parcel))}
    for line_number, row in enumerate(table.to_dict(orient="records"), start=2):
        code = str(row[id_column]).strip()
        parcel = parcels.get(code)
        if parcel is None:
            report.skipped += 1
            report.errors.append(f"Ligne {line_number} : parcelle inconnue {code!r} (importez d'abord la géométrie).")
            continue
        for variable in known:
            raw = row[variable]
            if raw is None or str(raw).strip() == "":
                continue
            spec = VARIABLE_CATALOG[variable]
            value = _parse_bool(raw) if spec.boolean else _parse_float(raw)
            if value is None or (
                (spec.minimum is not None and value < spec.minimum)
                or (spec.maximum is not None and value > spec.maximum)
            ):
                report.skipped += 1
                report.errors.append(f"Ligne {line_number}, {variable} : valeur invalide {raw!r}.")
                continue
            session.add(
                Observation(
                    parcel_id=parcel.id,
                    variable=variable,
                    value=value,
                    unit=spec.unit,
                    observed_start=observed_start,
                    observed_end=observed_end,
                    source=source.strip()[:500],
                    method="table_import",
                    is_demo=is_demo,
                )
            )
            report.created += 1
    session.flush()
    return report


def _parse_float(raw: Any) -> float | None:
    try:
        value = float(str(raw).replace(",", ".")) if isinstance(raw, str) else float(raw)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def add_observation(
    session: Session,
    parcel: Parcel,
    variable: str,
    value: float,
    *,
    source: str,
    method: str,
    unit: str | None = None,
    observed_start: date | None = None,
    observed_end: date | None = None,
    dataset_id: int | None = None,
    quality: dict[str, Any] | None = None,
    is_demo: bool = False,
) -> Observation:
    spec = VARIABLE_CATALOG.get(variable)
    if spec is not None:
        if (spec.minimum is not None and value < spec.minimum) or (
            spec.maximum is not None and value > spec.maximum
        ):
            raise ImportValidationError(f"{variable} = {value} hors de la plage physique admise.")
    observation = Observation(
        parcel_id=parcel.id,
        variable=variable,
        value=float(value),
        unit=unit or (spec.unit if spec else None),
        observed_start=observed_start,
        observed_end=observed_end,
        source=source[:500],
        method=method,
        dataset_id=dataset_id,
        quality=quality or {},
        is_demo=is_demo,
    )
    session.add(observation)
    return observation


# -------------------------------------------------------------------- datasets
def _relative_to_data(path: Path, data_dir: Path) -> str:
    return ensure_within(path, data_dir).relative_to(data_dir.resolve()).as_posix()


def dataset_path(dataset: Dataset, data_dir: Path) -> Path:
    return ensure_within(data_dir / dataset.path, data_dir)


def register_raster(
    session: Session,
    path: Path,
    data_dir: Path,
    *,
    name: str,
    source: str,
    product: str,
    satellite: bool,
    variable: str | None = None,
    categorical: bool = False,
    period_start: date | None = None,
    period_end: date | None = None,
    details: dict[str, Any] | None = None,
) -> Dataset:
    """Register a single-band GeoTIFF with explicit provenance."""
    try:
        _validate_layer_provenance(
            RasterLayer(
                name=name,
                path=path,
                source=source,
                satellite=satellite,
                product=product,
                categorical=categorical,
                period_start=period_start.isoformat() if period_start else None,
                period_end=period_end.isoformat() if period_end else None,
            )
        )
        metadata = inspect_raster(path)
    except GeospatialProcessingError as exc:
        raise ImportValidationError(str(exc)) from exc
    if metadata["band_count"] != 1:
        raise ImportValidationError("Le raster doit être mono-bande.")
    if metadata["crs"] is None:
        raise ImportValidationError("Le raster n'a pas de SCR déclaré.")
    if session.scalar(select(Dataset).where(Dataset.name == name)) is not None:
        raise ImportValidationError(f"Un jeu de données nommé {name!r} existe déjà.")
    resolution = metadata["resolution_native"]
    dataset = Dataset(
        name=name,
        kind="raster",
        variable=variable,
        source=source,
        product=product,
        satellite=satellite,
        categorical=categorical,
        path=_relative_to_data(path, data_dir),
        crs=metadata["crs"],
        resolution_m=_metric_resolution(metadata["crs"], resolution),
        period_start=period_start,
        period_end=period_end,
        checksum_sha256=sha256_file(path),
        details={**(details or {}), "raster": _json_safe_dict(metadata)},
    )
    session.add(dataset)
    session.flush()
    return dataset


def _metric_resolution(crs: str, resolution: list[float]) -> float | None:
    from pyproj import CRS

    parsed = CRS.from_user_input(crs)
    if parsed.is_projected and all(
        a.unit_name.lower() in {"metre", "meter"} for a in parsed.axis_info[:2]
    ):
        return float(max(resolution))
    return None  # geographic grids: metric size varies with latitude


def _json_safe_dict(data: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(data, default=str))


def register_vector(
    session: Session,
    path: Path,
    data_dir: Path,
    *,
    name: str,
    source: str,
    product: str,
    satellite: bool,
    variable: str | None = None,
) -> Dataset:
    """Register a reference vector layer (forests, villages, water sources…)."""
    if not source.strip() or not product.strip():
        raise ImportValidationError("source et product sont obligatoires.")
    if satellite and "nasa" not in source.casefold():
        raise ImportValidationError("Une couche satellitaire doit provenir d'une source NASA.")
    if "sentinel" in f"{source} {product} {name}".casefold():
        raise ImportValidationError("Les produits Sentinel ne sont pas autorisés dans ce projet.")
    frame = read_vector(path)
    if session.scalar(select(Dataset).where(Dataset.name == name)) is not None:
        raise ImportValidationError(f"Un jeu de données nommé {name!r} existe déjà.")
    dataset = Dataset(
        name=name,
        kind="vector",
        variable=variable,
        source=source,
        product=product,
        satellite=satellite,
        path=_relative_to_data(path, data_dir),
        crs=frame.crs.to_string(),
        checksum_sha256=sha256_file(path),
        details={
            "feature_count": int(len(frame)),
            "geometry_types": sorted({str(t) for t in frame.geom_type.dropna().unique()}),
            "bounds_wgs84": [float(v) for v in frame.to_crs(STORAGE_CRS).total_bounds],
        },
    )
    session.add(dataset)
    session.flush()
    return dataset


# ------------------------------------------------------------------------ demo
def load_demo_data(session: Session, settings: Any) -> dict[str, Any]:
    """Load the bundled pilot parcels and metrics, explicitly flagged as demo."""
    boundaries = settings.data_dir / "boundaries" / "sud_kivu_pilotes.geojson"
    metrics = settings.data_dir / "processed" / "parcelles_nasa_metrics.csv"
    if not boundaries.is_file() or not metrics.is_file():
        raise ImportValidationError("Les fichiers de démonstration du dépôt sont introuvables.")
    parcels_report = import_parcels(
        session, read_vector(boundaries), source=DEMO_SOURCE, is_demo=True
    )
    existing_demo = session.scalars(
        select(Observation).where(Observation.is_demo.is_(True), Observation.method == "table_import")
    ).all()
    for observation in existing_demo:  # make the demo load idempotent
        session.delete(observation)
    session.flush()
    metrics_report = import_metrics_table(session, metrics, source=DEMO_SOURCE, is_demo=True)
    return {"parcels": parcels_report.as_dict(), "metrics": metrics_report.as_dict()}
