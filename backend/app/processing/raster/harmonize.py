from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import numpy as np
import rasterio
from pyproj import CRS, Transformer
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.transform import from_origin
from rasterio.warp import reproject
from shapely.geometry import box
from shapely.ops import unary_union

from ..errors import GeospatialProcessingError


@dataclass(frozen=True)
class RasterLayer:
    """A raster and explicit provenance needed to process it safely."""

    name: str
    path: str | Path
    source: str
    satellite: bool
    product: str
    categorical: bool = False
    period_start: str | None = None
    period_end: str | None = None


@dataclass(frozen=True)
class RasterGrid:
    crs: CRS
    resolution_m: float
    transform: rasterio.Affine
    width: int
    height: int


def _validate_target_crs(target_crs: str) -> CRS:
    try:
        crs = CRS.from_user_input(target_crs)
    except Exception as exc:
        raise GeospatialProcessingError(
            f"Invalid target CRS {target_crs!r}: {exc}"
        ) from exc
    if not crs.is_projected:
        raise GeospatialProcessingError(
            f"Target CRS must be projected, not {crs.to_string()}."
        )
    if not crs.axis_info or any(
        axis.unit_name.lower() not in {"metre", "meter"}
        for axis in crs.axis_info[:2]
    ):
        raise GeospatialProcessingError(
            f"Target CRS must use metres, not {crs.to_string()}."
        )
    return crs


def _validate_layer_provenance(layer: RasterLayer) -> None:
    if not all(
        isinstance(value, str) and value.strip()
        for value in (layer.name, layer.source, layer.product)
    ):
        raise GeospatialProcessingError(
            "Each raster requires a non-empty path, name, source, and product."
        )
    if not isinstance(layer.path, (str, Path)) or not str(layer.path).strip():
        raise GeospatialProcessingError(
            "Each raster requires a non-empty path, name, source, and product."
        )
    if not isinstance(layer.satellite, bool):
        raise GeospatialProcessingError(
            f"Satellite provenance must be explicit for {layer.name}."
        )
    if layer.satellite:
        provenance = f"{layer.source} {layer.product} {layer.name}".casefold()
        if "nasa" not in layer.source.casefold():
            raise GeospatialProcessingError(
                f"Satellite layer {layer.name!r} is not identified as a NASA source."
            )
        if "sentinel" in provenance:
            raise GeospatialProcessingError(
                f"Satellite layer {layer.name!r} references Sentinel, which is not "
                "permitted for this project."
            )
    for label, value in (
        ("period_start", layer.period_start),
        ("period_end", layer.period_end),
    ):
        if value is not None:
            try:
                if not isinstance(value, str):
                    raise TypeError("date values must be strings")
                date.fromisoformat(value)
            except (TypeError, ValueError) as exc:
                raise GeospatialProcessingError(
                    f"{label} must be an ISO date (YYYY-MM-DD) for {layer.name}."
                ) from exc
    if (
        layer.period_start is not None
        and layer.period_end is not None
        and layer.period_start > layer.period_end
    ):
        raise GeospatialProcessingError(
            f"period_start is after period_end for {layer.name}."
        )


def _projected_resolution(
    dataset: rasterio.io.DatasetReader,
    boundary: gpd.GeoDataFrame,
    target_crs: CRS,
) -> tuple[float, float]:
    if dataset.crs is None:
        raise GeospatialProcessingError(
            f"Raster has no declared CRS: {dataset.name}"
        )
    source_boundary = unary_union(list(boundary.to_crs(dataset.crs).geometry))
    if not source_boundary.intersects(box(*dataset.bounds)):
        raise GeospatialProcessingError(
            f"Raster does not overlap the study area: {dataset.name}"
        )

    intersection = source_boundary.intersection(box(*dataset.bounds))
    point = intersection.representative_point()
    # ``DatasetReader.index`` returns (row, col). ``DatasetReader.xy`` is used for
    # pixel-corner coordinates because ``Affine @ tuple`` is unavailable in affine<3.
    row, col = dataset.index(point.x, point.y)
    col = min(max(col, 0), dataset.width - 1)
    row = min(max(row, 0), dataset.height - 1)
    x0, y0 = dataset.xy(row, col, offset="ul")
    x1, y1 = dataset.xy(row, col + 1, offset="ul")
    x2, y2 = dataset.xy(row + 1, col, offset="ul")
    transformer = Transformer.from_crs(dataset.crs, target_crs, always_xy=True)
    tx0, ty0 = transformer.transform(x0, y0)
    tx1, ty1 = transformer.transform(x1, y1)
    tx2, ty2 = transformer.transform(x2, y2)
    x_resolution = math.hypot(tx1 - tx0, ty1 - ty0)
    y_resolution = math.hypot(tx2 - tx0, ty2 - ty0)
    if not all(math.isfinite(value) and value > 0 for value in (x_resolution, y_resolution)):
        raise GeospatialProcessingError(
            f"Could not determine pixel resolution for {dataset.name}."
        )
    return x_resolution, y_resolution


def _make_grid(
    bounds: tuple[float, float, float, float],
    target_crs: CRS,
    resolution_m: float,
) -> RasterGrid:
    min_x, min_y, max_x, max_y = bounds
    left = math.floor(min_x / resolution_m) * resolution_m
    right = math.ceil(max_x / resolution_m) * resolution_m
    bottom = math.floor(min_y / resolution_m) * resolution_m
    top = math.ceil(max_y / resolution_m) * resolution_m
    width = max(1, int(round((right - left) / resolution_m)))
    height = max(1, int(round((top - bottom) / resolution_m)))
    return RasterGrid(
        crs=target_crs,
        resolution_m=resolution_m,
        transform=from_origin(left, top, resolution_m, resolution_m),
        width=width,
        height=height,
    )


def harmonize_rasters(
    layers: Iterable[RasterLayer],
    study_area: gpd.GeoDataFrame,
    output_dir: str | Path,
    target_crs: str = "EPSG:32735",
    target_resolution_m: float | None = None,
) -> tuple[dict[str, Path], dict[str, object]]:
    """Reproject and clip rasters onto one grid without upsampling the coarsest input.

    Continuous layers are averaged onto coarser cells; categorical layers use nearest
    neighbour. The default grid uses the coarsest native pixel size observed locally.
    """
    layer_list = list(layers)
    if not layer_list:
        raise GeospatialProcessingError("At least one raster layer is required.")
    if study_area.empty or study_area.crs is None:
        raise GeospatialProcessingError(
            "Study area must be non-empty and have a declared CRS."
        )

    target = _validate_target_crs(target_crs)
    if CRS.from_user_input(study_area.crs) != target:
        study_area = study_area.to_crs(target)
    geometry = unary_union(list(study_area.geometry))
    if geometry.is_empty:
        raise GeospatialProcessingError("Study-area geometry is empty.")

    source_infos: list[tuple[RasterLayer, tuple[float, float]]] = []
    names: set[str] = set()
    output_names: dict[str, str] = {}
    normalized_names: set[str] = set()
    for layer in layer_list:
        _validate_layer_provenance(layer)
        if layer.name in names:
            raise GeospatialProcessingError(f"Duplicate raster layer name: {layer.name}")
        names.add(layer.name)
        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", layer.name).strip("._")
        if not safe_name:
            raise GeospatialProcessingError(
                f"Raster layer name cannot form an output filename: {layer.name!r}"
            )
        if safe_name in normalized_names:
            raise GeospatialProcessingError(
                f"Raster layer names collide after filename normalization: "
                f"{layer.name!r}"
            )
        normalized_names.add(safe_name)
        output_names[layer.name] = safe_name
        raster_path = Path(layer.path)
        if not raster_path.is_file():
            raise GeospatialProcessingError(
                f"Raster does not exist for layer {layer.name!r}: {raster_path}"
            )
        try:
            with rasterio.open(raster_path) as dataset:
                if dataset.count != 1:
                    raise GeospatialProcessingError(
                        f"Layer {layer.name!r} has {dataset.count} bands; "
                        "provide a single-band raster for each layer."
                    )
                resolution = _projected_resolution(dataset, study_area, target)
        except rasterio.errors.RasterioError as exc:
            raise GeospatialProcessingError(
                f"Could not open raster for layer {layer.name!r}: {exc}"
            ) from exc
        source_infos.append((layer, resolution))

    coarsest_native_m = max(max(resolution) for _, resolution in source_infos)
    try:
        resolution_m = (
            coarsest_native_m
            if target_resolution_m is None
            else float(target_resolution_m)
        )
    except (TypeError, ValueError) as exc:
        raise GeospatialProcessingError(
            "target_resolution_m must be a positive finite number."
        ) from exc
    if not math.isfinite(resolution_m) or resolution_m <= 0:
        raise GeospatialProcessingError(
            "target_resolution_m must be a positive finite number."
        )
    if resolution_m + 1e-8 < coarsest_native_m:
        raise GeospatialProcessingError(
            f"Requested grid ({resolution_m:g} m) is finer than the coarsest "
            f"input pixel ({coarsest_native_m:g} m); this would imply false detail."
        )

    grid = _make_grid(geometry.bounds, target, resolution_m)
    study_area_mask = geometry_mask(
        [geometry.__geo_interface__],
        out_shape=(grid.height, grid.width),
        transform=grid.transform,
        all_touched=False,
        invert=True,
    )
    destination_dir = Path(output_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    metadata: dict[str, object] = {
        "target_crs": target.to_string(),
        "target_resolution_m": resolution_m,
        "grid_width": grid.width,
        "grid_height": grid.height,
        "boundary_bounds": list(geometry.bounds),
        "layers": {},
    }

    for layer, native_resolution in source_infos:
        safe_name = output_names[layer.name]
        destination_path = destination_dir / f"{safe_name}.tif"
        nodata = -9999.0
        resampling = (
            Resampling.nearest if layer.categorical else Resampling.average
        )
        destination = np.full(
            (grid.height, grid.width), nodata, dtype=np.float32
        )
        with rasterio.open(layer.path) as source:
            source_scale = float(source.scales[0])
            source_offset = float(source.offsets[0])
            source_unit = source.units[0]
            reproject(
                source=rasterio.band(source, 1),
                destination=destination,
                src_transform=source.transform,
                src_crs=source.crs,
                src_nodata=source.nodata,
                dst_transform=grid.transform,
                dst_crs=grid.crs,
                dst_nodata=nodata,
                resampling=resampling,
                init_dest_nodata=True,
            )

        destination[~study_area_mask] = nodata
        profile = {
            "driver": "GTiff",
            "height": grid.height,
            "width": grid.width,
            "count": 1,
            "dtype": "float32",
            "crs": grid.crs,
            "transform": grid.transform,
            "nodata": nodata,
            "compress": "deflate",
            "tiled": True,
        }
        with rasterio.open(destination_path, "w", **profile) as output:
            output.write(destination, 1)
            output.set_band_description(1, layer.name)
            output.scales = (source_scale,)
            output.offsets = (source_offset,)
            output.update_tags(
                source=layer.source,
                product=layer.product,
                satellite=str(layer.satellite).lower(),
                period_start=layer.period_start or "",
                period_end=layer.period_end or "",
                source_unit=source_unit or "",
                source_scale=str(source_scale),
                source_offset=str(source_offset),
                source_resolution_x_m=f"{native_resolution[0]:.6g}",
                source_resolution_y_m=f"{native_resolution[1]:.6g}",
                output_resolution_m=f"{resolution_m:.6g}",
                resampling=resampling.name,
                resolution_note=(
                    "Output is not finer than the coarsest source pixel; "
                    "resampling does not increase native information."
                ),
            )
        with rasterio.open(destination_path) as check:
            if (
                check.crs != grid.crs
                or check.transform != grid.transform
                or check.width != grid.width
                or check.height != grid.height
            ):
                raise GeospatialProcessingError(
                    f"Prepared raster failed grid verification: {destination_path}"
                )
            checked_values = check.read(1, masked=True)
            valid_pixel_count = int(checked_values.count())
            if valid_pixel_count == 0:
                raise GeospatialProcessingError(
                    f"Prepared raster has no valid pixels in the study area: "
                    f"{destination_path}"
                )
            if np.any(destination[~study_area_mask] != nodata):
                raise GeospatialProcessingError(
                    f"Prepared raster contains values outside the study area: "
                    f"{destination_path}"
                )
        outputs[layer.name] = destination_path
        metadata["layers"][layer.name] = {
            "source_path": str(layer.path),
            "source": layer.source,
            "product": layer.product,
            "satellite": layer.satellite,
            "period_start": layer.period_start,
            "period_end": layer.period_end,
            "unit": source_unit,
            "scale": source_scale,
            "offset": source_offset,
            "categorical": layer.categorical,
            "native_resolution_m": list(native_resolution),
            "output_resolution_m": resolution_m,
            "resampling": resampling.name,
            "output_path": str(destination_path),
            "valid_pixel_count": valid_pixel_count,
        }

    return outputs, metadata
