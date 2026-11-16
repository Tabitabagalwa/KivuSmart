from __future__ import annotations

from pathlib import Path

import geopandas as gpd
from shapely import make_valid
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
from shapely.ops import unary_union

from ..errors import GeospatialProcessingError


def _polygon_parts(geometry: object) -> list[Polygon]:
    if isinstance(geometry, Polygon):
        return [geometry]
    if isinstance(geometry, MultiPolygon):
        return list(geometry.geoms)
    if isinstance(geometry, GeometryCollection):
        return [
            polygon
            for part in geometry.geoms
            for polygon in _polygon_parts(part)
        ]
    return []


def load_study_area(
    boundary_path: str | Path,
    target_crs: str = "EPSG:32735",
    study_area_name: str = "Kalehe",
    boundary_source: str | None = None,
) -> gpd.GeoDataFrame:
    """Load an authoritative polygon boundary and return a valid dissolved area.

    The source CRS must be declared in the file; it is never inferred from coordinates.
    """
    path = Path(boundary_path)
    if not path.is_file():
        raise GeospatialProcessingError(
            f"Study-area boundary does not exist: {path}. "
            "Provide an authoritative Kalehe administrative boundary."
        )

    try:
        source = gpd.read_file(path)
    except Exception as exc:
        raise GeospatialProcessingError(
            f"Could not read study-area boundary {path}: {exc}"
        ) from exc

    if source.empty:
        raise GeospatialProcessingError(f"Study-area boundary is empty: {path}")
    if source.crs is None:
        raise GeospatialProcessingError(
            f"Study-area boundary has no declared CRS: {path}. "
            "Assign its verified source CRS before processing."
        )

    try:
        target = source.to_crs(target_crs)
    except Exception as exc:
        raise GeospatialProcessingError(
            f"Could not transform the boundary to {target_crs}: {exc}"
        ) from exc

    polygon_geometries: list[Polygon] = []
    for geometry in target.geometry:
        if geometry is None or geometry.is_empty:
            continue
        polygon_geometries.extend(_polygon_parts(make_valid(geometry)))

    if not polygon_geometries:
        raise GeospatialProcessingError(
            f"Boundary contains no usable polygon geometry: {path}"
        )

    dissolved = make_valid(unary_union(polygon_geometries))
    dissolved_polygons = _polygon_parts(dissolved)
    if not dissolved_polygons:
        raise GeospatialProcessingError(
            f"Boundary could not be repaired into polygon geometry: {path}"
        )

    properties: dict[str, list[str]] = {"study_area": [study_area_name]}
    if boundary_source is not None:
        properties["boundary_source"] = [boundary_source]
    return gpd.GeoDataFrame(
        properties,
        geometry=[unary_union(dissolved_polygons)],
        crs=target_crs,
    )


def write_study_area(
    study_area: gpd.GeoDataFrame,
    output_path: str | Path,
) -> Path:
    """Write a prepared boundary, creating only the requested output directory."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    study_area.to_file(destination, driver="GeoJSON", index=False)
    return destination
