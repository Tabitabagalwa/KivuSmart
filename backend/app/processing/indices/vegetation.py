from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio

from ..errors import GeospatialProcessingError


def calculate_ndvi(
    red_path: str | Path,
    nir_path: str | Path,
    output_path: str | Path,
    source: str,
    product: str,
) -> Path:
    """Calculate NDVI from pre-aligned red and NIR single-band rasters.

    Inputs must already share one grid. Provenance is required and Sentinel-derived
    satellite inputs are rejected.
    """
    if not source.strip() or not product.strip():
        raise GeospatialProcessingError(
            "NDVI calculation requires explicit source and product metadata."
        )
    if "nasa" not in source.casefold():
        raise GeospatialProcessingError(
            "Satellite-derived NDVI inputs must be identified as a NASA source."
        )
    if "sentinel" in f"{source} {product}".casefold():
        raise GeospatialProcessingError(
            "Sentinel-derived NDVI is not permitted for this project."
        )

    with rasterio.open(red_path) as red, rasterio.open(nir_path) as nir:
        if red.count != 1 or nir.count != 1:
            raise GeospatialProcessingError("NDVI inputs must be single-band rasters.")
        if (
            red.crs != nir.crs
            or red.transform != nir.transform
            or red.width != nir.width
            or red.height != nir.height
        ):
            raise GeospatialProcessingError(
                "Red and NIR rasters must have identical CRS, transform, and dimensions. "
                "Harmonize them before calculating NDVI."
            )
        if red.crs is None:
            raise GeospatialProcessingError("NDVI inputs have no declared CRS.")

        red_values = red.read(1, masked=True).astype(np.float32)
        nir_values = nir.read(1, masked=True).astype(np.float32)
        red_values = red_values * float(red.scales[0]) + float(red.offsets[0])
        nir_values = nir_values * float(nir.scales[0]) + float(nir.offsets[0])
        denominator = nir_values + red_values
        invalid = (
            np.ma.getmaskarray(red_values)
            | np.ma.getmaskarray(nir_values)
            | ~np.isfinite(red_values.data)
            | ~np.isfinite(nir_values.data)
            | ~np.isfinite(denominator.filled(np.nan))
            | (denominator.filled(0) == 0)
        )
        values = np.full((red.height, red.width), -9999.0, dtype=np.float32)
        valid = ~invalid
        values[valid] = (
            (nir_values.data[valid] - red_values.data[valid])
            / denominator.data[valid]
        )
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        profile = red.profile.copy()
        profile.update(dtype="float32", nodata=-9999.0, compress="deflate")
        with rasterio.open(destination, "w", **profile) as output:
            output.write(values, 1)
            output.set_band_description(1, "NDVI")
            output.scales = (1.0,)
            output.offsets = (0.0,)
            output.update_tags(
                source=source,
                product=product,
                unit="dimensionless",
                formula="(NIR - red) / (NIR + red)",
                resolution_note="Computed on the input grid; no spatial detail is added.",
            )
    return destination
