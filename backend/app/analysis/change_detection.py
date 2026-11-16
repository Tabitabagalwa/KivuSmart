"""Bi-temporal raster change detection (e.g. NDVI before/after).

Both rasters must already share one grid (use ``harmonize_rasters``): this module
never resamples, so the reported change cannot contain interpolation artefacts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from pyproj import CRS

from ..processing.errors import GeospatialProcessingError

NODATA_FLOAT = -9999.0
NODATA_CLASS = -128
CLASS_LABELS = {-1: "perte", 0: "stable", 1: "gain"}


def _read_scaled(dataset: rasterio.io.DatasetReader) -> np.ma.MaskedArray:
    values = dataset.read(1, masked=True).astype(np.float64)
    values = values * float(dataset.scales[0]) + float(dataset.offsets[0])
    return np.ma.masked_invalid(values)


def _pixel_area_m2(dataset: rasterio.io.DatasetReader) -> float | None:
    crs = CRS.from_user_input(dataset.crs)
    if crs.is_projected and all(
        a.unit_name.lower() in {"metre", "meter"} for a in crs.axis_info[:2]
    ):
        return abs(dataset.transform.a * dataset.transform.e)
    return None


def raster_change(
    before_path: str | Path,
    after_path: str | Path,
    output_dir: str | Path,
    *,
    threshold: float,
    name: str = "change",
) -> dict[str, Any]:
    """Write a difference raster (after - before) and a loss/stable/gain class raster."""
    if not np.isfinite(threshold) or threshold < 0:
        raise GeospatialProcessingError("Le seuil de changement doit être un nombre positif.")
    with rasterio.open(before_path) as before, rasterio.open(after_path) as after:
        if before.count != 1 or after.count != 1:
            raise GeospatialProcessingError("Les rasters comparés doivent être mono-bande.")
        if before.crs is None or after.crs is None:
            raise GeospatialProcessingError("Les rasters comparés doivent déclarer un SCR.")
        if (
            before.crs != after.crs
            or before.transform != after.transform
            or before.shape != after.shape
        ):
            raise GeospatialProcessingError(
                "Les rasters n'ont pas la même grille (SCR, transformation, dimensions). "
                "Harmonisez-les avant l'analyse de changement."
            )
        first = _read_scaled(before)
        second = _read_scaled(after)
        mask = np.ma.getmaskarray(first) | np.ma.getmaskarray(second)
        difference = np.where(mask, NODATA_FLOAT, second.filled(0) - first.filled(0))
        classes = np.full(difference.shape, NODATA_CLASS, dtype=np.int16)
        valid = ~mask
        classes[valid & (difference <= -threshold)] = -1
        classes[valid & (np.abs(difference) < threshold)] = 0
        classes[valid & (difference >= threshold)] = 1
        pixel_area = _pixel_area_m2(before)
        profile = before.profile.copy()

    valid_count = int(valid.sum())
    if valid_count == 0:
        raise GeospatialProcessingError("Aucun pixel valide commun aux deux dates.")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    diff_path = destination / f"{name}_difference.tif"
    class_path = destination / f"{name}_classes.tif"
    profile.update(driver="GTiff", count=1, compress="deflate")
    with rasterio.open(diff_path, "w", **{**profile, "dtype": "float32", "nodata": NODATA_FLOAT}) as out:
        out.write(difference.astype(np.float32), 1)
        out.set_band_description(1, f"{name}: after - before")
        out.scales, out.offsets = (1.0,), (0.0,)
        out.update_tags(threshold=str(threshold), method="after - before, no resampling")
    with rasterio.open(class_path, "w", **{**profile, "dtype": "int16", "nodata": NODATA_CLASS}) as out:
        out.write(classes, 1)
        out.set_band_description(1, f"{name}: -1 perte, 0 stable, 1 gain")
        out.scales, out.offsets = (1.0,), (0.0,)
        out.update_tags(classes="-1=perte,0=stable,1=gain", threshold=str(threshold))

    valid_diff = difference[valid]
    breakdown = {}
    for value, label in CLASS_LABELS.items():
        count = int((classes == value).sum())
        breakdown[label] = {
            "pixels": count,
            "fraction": round(count / valid_count, 4),
            "area_ha": round(count * pixel_area / 10_000, 4) if pixel_area else None,
        }
    return {
        "difference_path": str(diff_path),
        "classes_path": str(class_path),
        "threshold": threshold,
        "valid_pixel_count": valid_count,
        "mean_difference": float(valid_diff.mean()),
        "median_difference": float(np.median(valid_diff)),
        "min_difference": float(valid_diff.min()),
        "max_difference": float(valid_diff.max()),
        "classes": breakdown,
        "area_note": None if pixel_area else (
            "SCR géographique : surfaces non calculées (pixels de taille variable)."
        ),
    }
