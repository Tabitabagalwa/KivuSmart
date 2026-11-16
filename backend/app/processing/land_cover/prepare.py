from __future__ import annotations

from pathlib import Path

import rasterio

from ..errors import GeospatialProcessingError


def validate_categorical_layer(path: str | Path) -> dict[str, object]:
    """Check that a prepared land-cover raster is single-band and has a CRS."""
    raster_path = Path(path)
    if not raster_path.is_file():
        raise GeospatialProcessingError(
            f"Land-cover raster does not exist: {raster_path}"
        )
    with rasterio.open(raster_path) as dataset:
        if dataset.count != 1:
            raise GeospatialProcessingError(
                f"Land-cover raster must be single-band: {raster_path}"
            )
        if dataset.crs is None:
            raise GeospatialProcessingError(
                f"Land-cover raster has no declared CRS: {raster_path}"
            )
        return {
            "path": str(raster_path),
            "crs": dataset.crs.to_string(),
            "resolution": [abs(dataset.transform.a), abs(dataset.transform.e)],
            "width": dataset.width,
            "height": dataset.height,
            "nodata": dataset.nodata,
            "resampling_requirement": "nearest",
        }
