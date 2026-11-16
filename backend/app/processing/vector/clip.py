from __future__ import annotations

from pathlib import Path

import geopandas as gpd
from shapely.ops import unary_union

from ..errors import GeospatialProcessingError


def clip_vector_layer(
    vector_path: str | Path,
    study_area: gpd.GeoDataFrame,
    output_path: str | Path | None = None,
) -> gpd.GeoDataFrame:
    """Reproject and intersect a vector layer using the study-area CRS."""
    try:
        layer = gpd.read_file(vector_path)
    except Exception as exc:
        raise GeospatialProcessingError(
            f"Could not read vector layer {vector_path}: {exc}"
        ) from exc

    if layer.crs is None:
        raise GeospatialProcessingError(
            f"Vector layer has no declared CRS: {vector_path}"
        )
    if study_area.crs is None:
        raise GeospatialProcessingError("Study area has no declared CRS.")
    if layer.empty:
        raise GeospatialProcessingError(f"Vector layer is empty: {vector_path}")

    layer = layer.to_crs(study_area.crs)
    boundary = unary_union(list(study_area.geometry))
    clipped = layer.copy()
    clipped.geometry = clipped.geometry.intersection(boundary)
    clipped = clipped.loc[~clipped.geometry.is_empty].copy()
    clipped = clipped.loc[clipped.geometry.notna()].copy()

    if clipped.empty:
        raise GeospatialProcessingError(
            f"Vector layer does not intersect the study area: {vector_path}"
        )

    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        driver = "GeoJSON" if destination.suffix.lower() in {".json", ".geojson"} else None
        clipped.to_file(destination, driver=driver, index=False)

    return clipped
