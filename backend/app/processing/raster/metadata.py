from __future__ import annotations

from pathlib import Path
from typing import Any

import rasterio

from ..errors import GeospatialProcessingError


def inspect_raster(path: str | Path) -> dict[str, Any]:
    """Return observed GeoTIFF metadata without guessing missing values."""
    raster_path = Path(path)
    if not raster_path.is_file():
        raise GeospatialProcessingError(f"Raster does not exist: {raster_path}")

    try:
        with rasterio.open(raster_path) as dataset:
            return {
                "path": str(raster_path),
                "driver": dataset.driver,
                "width": dataset.width,
                "height": dataset.height,
                "band_count": dataset.count,
                "dtypes": list(dataset.dtypes),
                "crs": dataset.crs.to_string() if dataset.crs else None,
                "bounds": [
                    dataset.bounds.left,
                    dataset.bounds.bottom,
                    dataset.bounds.right,
                    dataset.bounds.top,
                ],
                "resolution_native": [
                    abs(dataset.transform.a),
                    abs(dataset.transform.e),
                ],
                "nodata": dataset.nodata,
                "units": list(dataset.units),
                "scales": list(dataset.scales),
                "offsets": list(dataset.offsets),
                "descriptions": list(dataset.descriptions),
                "tags": dataset.tags(),
            }
    except rasterio.errors.RasterioError as exc:
        raise GeospatialProcessingError(
            f"Could not inspect raster {raster_path}: {exc}"
        ) from exc
