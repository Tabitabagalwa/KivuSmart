from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.mask import mask

from .errors import GeospatialProcessingError


def extract_zonal_statistics(
    raster_paths: dict[str, str | Path],
    zones: gpd.GeoDataFrame,
    zone_id_field: str | None = None,
) -> gpd.GeoDataFrame:
    """Summarize valid raster cells per zone without inventing missing values."""
    if zones.empty or zones.crs is None:
        raise GeospatialProcessingError(
            "Zones must be non-empty and have a declared CRS."
        )
    if not raster_paths:
        raise GeospatialProcessingError("At least one raster is required.")
    if zone_id_field is not None and zone_id_field not in zones.columns:
        raise GeospatialProcessingError(
            f"Zone ID field is not present: {zone_id_field}"
        )

    results: list[dict[str, object]] = []
    for zone_index, zone in zones.iterrows():
        geometry = zone.geometry
        if geometry is None or geometry.is_empty:
            continue
        zone_id = (
            zone[zone_id_field] if zone_id_field is not None else zone_index
        )
        for layer_name, raster_path in raster_paths.items():
            with rasterio.open(raster_path) as dataset:
                if dataset.crs is None:
                    raise GeospatialProcessingError(
                        f"Raster has no declared CRS: {raster_path}"
                    )
                if dataset.count != 1:
                    raise GeospatialProcessingError(
                        f"Zonal statistics require one band per raster: {raster_path}"
                    )
                projected_geometry = (
                    gpd.GeoSeries([geometry], crs=zones.crs)
                    .to_crs(dataset.crs)
                    .iloc[0]
                )
                try:
                    values, _ = mask(
                        dataset,
                        [projected_geometry.__geo_interface__],
                        crop=True,
                        filled=False,
                    )
                except ValueError as exc:
                    raise GeospatialProcessingError(
                        f"Zone {zone_id!r} does not overlap raster {layer_name!r}."
                    ) from exc

                band = values[0]
                valid = np.asarray(band.compressed(), dtype=np.float64)
                valid = valid[np.isfinite(valid)]
                if dataset.nodata is not None:
                    valid = valid[valid != dataset.nodata]
                if valid.size == 0:
                    raise GeospatialProcessingError(
                        f"Zone {zone_id!r} has no valid pixels in {layer_name!r}."
                    )
                scale = float(dataset.scales[0])
                offset = float(dataset.offsets[0])
                valid = valid * scale + offset
                results.append(
                    {
                        "zone_id": zone_id,
                        "layer": layer_name,
                        "valid_pixel_count": int(valid.size),
                        "mean": float(valid.mean()),
                        "std": float(valid.std()),
                        "min": float(valid.min()),
                        "max": float(valid.max()),
                        "crs": dataset.crs.to_string(),
                        "unit": dataset.units[0],
                        "scale": scale,
                        "offset": offset,
                        "resolution_x": abs(dataset.transform.a),
                        "resolution_y": abs(dataset.transform.e),
                    }
                )

    return gpd.GeoDataFrame(results)


def write_zonal_statistics(
    statistics: gpd.GeoDataFrame,
    output_path: str | Path,
) -> Path:
    """Write non-spatial zonal summaries as a CSV table."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    statistics.drop(columns="geometry", errors="ignore").to_csv(
        destination, index=False
    )
    return destination
