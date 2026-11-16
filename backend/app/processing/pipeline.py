from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .errors import GeospatialProcessingError
from .features import extract_zonal_statistics, write_zonal_statistics
from .raster.harmonize import RasterLayer, harmonize_rasters
from .vector.boundaries import load_study_area, write_study_area

DEFAULT_TARGET_CRS = "EPSG:32735"


@dataclass(frozen=True)
class RasterLayerInput:
    """Serializable raster input definition for the geospatial pipeline."""

    name: str
    path: str
    source: str
    satellite: bool
    product: str
    categorical: bool = False
    period_start: str | None = None
    period_end: str | None = None

    def as_layer(self) -> RasterLayer:
        return RasterLayer(
            name=self.name,
            path=self.path,
            source=self.source,
            satellite=self.satellite,
            product=self.product,
            categorical=self.categorical,
            period_start=self.period_start,
            period_end=self.period_end,
        )


def run_geospatial_pipeline(
    boundary_path: str | Path,
    boundary_source: str,
    raster_layers: list[RasterLayerInput],
    output_dir: str | Path,
    target_crs: str = DEFAULT_TARGET_CRS,
    target_resolution_m: float | None = None,
) -> dict[str, object]:
    """Validate, project, align, clip, and summarize supplied study-area layers.

    An official boundary and explicit raster provenance are required. The pipeline
    does not fetch data, infer missing metadata, or implement recommendations.
    """
    if boundary_path is None or not str(boundary_path).strip():
        raise GeospatialProcessingError(
            "No Kalehe boundary was supplied. Configure an authoritative boundary file "
            "before running geospatial processing."
        )
    if not boundary_source or not boundary_source.strip():
        raise GeospatialProcessingError(
            "Boundary provenance is required; provide boundary_source explicitly."
        )
    if not raster_layers:
        raise GeospatialProcessingError(
            "No raster layers were supplied; provide sourced input rasters."
        )

    destination = Path(output_dir)
    boundary = load_study_area(
        boundary_path,
        target_crs=target_crs,
        boundary_source=boundary_source,
    )
    boundary_output = write_study_area(
        boundary, destination / "boundaries" / "kalehe_study_area.geojson"
    )
    verified_boundary = load_study_area(
        boundary_output,
        target_crs=target_crs,
        boundary_source=boundary_source,
    )
    if (
        verified_boundary.crs != boundary.crs
        or not verified_boundary.geometry.iloc[0].equals(
            boundary.geometry.iloc[0]
        )
    ):
        raise GeospatialProcessingError(
            "Written study-area boundary failed its CRS or geometry verification."
        )

    # Validate all inputs and derive the target grid before writing intermediate files.
    outputs, raster_report = harmonize_rasters(
        [layer.as_layer() for layer in raster_layers],
        verified_boundary,
        destination / "rasters",
        target_crs=target_crs,
        target_resolution_m=target_resolution_m,
    )

    statistics = extract_zonal_statistics(outputs, verified_boundary)
    statistics_path = write_zonal_statistics(
        statistics, destination / "environmental_zonal_statistics.csv"
    )

    report: dict[str, object] = {
        "study_area": "Kalehe",
        "boundary_source_path": str(Path(boundary_path)),
        "boundary_source": boundary_source,
        "boundary_output_path": str(boundary_output),
        "boundary_crs": boundary.crs.to_string(),
        "boundary_feature_count": len(verified_boundary),
        "target_crs": target_crs,
        "target_resolution_m": raster_report["target_resolution_m"],
        "raster_layers": raster_report["layers"],
        "zonal_statistics_path": str(statistics_path),
        "notes": [
            "Raster resolution is never made finer than the coarsest native input.",
            "Resampling aligns data but does not increase native spatial information.",
            "Source temporal coverage is recorded only when explicitly supplied.",
            "This pipeline prepares data and does not issue restoration recommendations.",
        ],
    }
    report_path = destination / "processing_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report


def load_raster_layer_config(path: str | Path) -> list[RasterLayerInput]:
    """Read an explicit JSON raster manifest; reject omitted provenance fields."""
    manifest_path = Path(path)
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GeospatialProcessingError(
            f"Could not read raster manifest {manifest_path}: {exc}"
        ) from exc
    items = raw.get("rasters")
    if not isinstance(items, list):
        raise GeospatialProcessingError(
            "Raster manifest must contain a 'rasters' array."
        )
    try:
        layers = [RasterLayerInput(**item) for item in items]
    except (TypeError, ValueError) as exc:
        raise GeospatialProcessingError(
            f"Raster manifest entry is invalid: {exc}"
        ) from exc
    if not layers:
        raise GeospatialProcessingError("Raster manifest contains no raster entries.")
    project_root = manifest_path.resolve().parent.parent
    return [
        layer
        if Path(layer.path).is_absolute()
        else RasterLayerInput(
            **{
                **layer.__dict__,
                "path": str(project_root / layer.path),
            }
        )
        for layer in layers
    ]


def run_configured_pipeline(
    study_area_config_path: str | Path,
    raster_manifest_path: str | Path,
    output_dir: str | Path,
) -> dict[str, object]:
    """Run the workflow from the project JSON configuration files."""
    config_path = Path(study_area_config_path)
    try:
        configuration = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GeospatialProcessingError(
            f"Could not read study-area configuration {config_path}: {exc}"
        ) from exc

    configured_boundary = configuration.get("boundary_path")
    configured_boundary_source = configuration.get("boundary_source")
    if not configured_boundary or not configured_boundary_source:
        raise GeospatialProcessingError(
            f"No authoritative boundary is configured in {config_path}. "
            "Set boundary_path and boundary_source before processing."
        )
    boundary_path = Path(configured_boundary)
    if not boundary_path.is_absolute():
        boundary_path = config_path.resolve().parent.parent / boundary_path

    return run_geospatial_pipeline(
        boundary_path=boundary_path,
        boundary_source=configured_boundary_source,
        raster_layers=load_raster_layer_config(raster_manifest_path),
        output_dir=output_dir,
        target_crs=configuration.get("target_crs", DEFAULT_TARGET_CRS),
        target_resolution_m=configuration.get("target_resolution_m"),
    )
