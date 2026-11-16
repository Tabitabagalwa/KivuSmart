from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from pyproj import CRS

from ..errors import GeospatialProcessingError


def _write_derived_raster(
    template: rasterio.io.DatasetReader,
    output_path: Path,
    values: np.ndarray,
    description: str,
    nodata: float,
) -> None:
    profile = template.profile.copy()
    profile.update(
        driver="GTiff",
        dtype="float32",
        count=1,
        nodata=nodata,
        compress="deflate",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, "w", **profile) as output:
        output.write(values.astype(np.float32), 1)
        output.set_band_description(1, description)
        output.scales = (1.0,)
        output.offsets = (0.0,)
        output.update_tags(
            derivation="Derived from the supplied elevation raster.",
            unit="percent" if description == "slope_percent" else "degrees",
            resolution_note="Derived at source grid resolution; no new detail is created.",
        )


def derive_terrain_attributes(
    elevation_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Derive slope in percent and aspect in degrees from a projected DEM."""
    output_dir = Path(output_dir)
    with rasterio.open(elevation_path) as elevation:
        if elevation.count != 1 or elevation.crs is None:
            raise GeospatialProcessingError(
                "Elevation input must be a single-band raster with a declared CRS."
            )
        crs = CRS.from_user_input(elevation.crs)
        if not crs.is_projected or any(
            axis.unit_name.lower() not in {"metre", "meter"}
            for axis in crs.axis_info[:2]
        ):
            raise GeospatialProcessingError(
                "Terrain derivatives require a projected DEM whose units are metres."
            )
        pixel_x = abs(elevation.transform.a)
        pixel_y = abs(elevation.transform.e)
        if pixel_x <= 0 or pixel_y <= 0:
            raise GeospatialProcessingError("DEM pixel size must be positive.")
        if not np.isclose(pixel_x, pixel_y, rtol=1e-4):
            raise GeospatialProcessingError(
                "Terrain derivatives require square pixels after grid harmonization."
            )

        source = elevation.read(1, masked=True).astype(np.float64)
        dem = source.filled(np.nan)
        dem = dem * float(elevation.scales[0]) + float(elevation.offsets[0])
        if dem.shape[0] < 2 or dem.shape[1] < 2:
            raise GeospatialProcessingError(
                "Elevation raster must contain at least 2 rows and 2 columns."
            )
        if np.all(~np.isfinite(dem)):
            raise GeospatialProcessingError("Elevation raster has no valid pixels.")
        dz_dy, dz_dx = np.gradient(dem, pixel_y, pixel_x)
        slope_percent = np.hypot(dz_dx, dz_dy) * 100.0
        aspect_degrees = np.degrees(np.arctan2(-dz_dx, dz_dy)) % 360.0
        valid = np.isfinite(dem) & np.isfinite(slope_percent) & np.isfinite(aspect_degrees)
        nodata = -9999.0
        slope_percent[~valid] = nodata
        aspect_degrees[~valid] = nodata

        slope_path = output_dir / "slope_percent.tif"
        aspect_path = output_dir / "aspect_degrees.tif"
        _write_derived_raster(
            elevation, slope_path, slope_percent, "slope_percent", nodata
        )
        _write_derived_raster(
            elevation, aspect_path, aspect_degrees, "aspect_degrees", nodata
        )
    return {"slope_percent": slope_path, "aspect_degrees": aspect_path}
