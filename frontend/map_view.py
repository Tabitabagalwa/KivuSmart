from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import folium
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import streamlit as st
from folium.raster_layers import ImageOverlay
from rasterio.mask import mask as mask_raster
from rasterio.transform import array_bounds
from rasterio.warp import calculate_default_transform, reproject
from rasterio.enums import Resampling
from shapely.geometry import shape
from shapely.ops import unary_union
from streamlit_folium import st_folium

from KivuSmart.backend.app.processing.errors import GeospatialProcessingError
from KivuSmart.backend.app.processing.features import extract_zonal_statistics


PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Mode simulation : KIVU_SIMULATION=1 lit config/simulation/ (données synthétiques
# produites par scripts/generer_donnees_simulees.py). Sans la variable, la
# configuration réelle (config/) est utilisée, sans aucune modification.
SIMULATION_MODE = os.getenv("KIVU_SIMULATION", "").strip().lower() in {"1", "true", "oui", "yes"}
CONFIG_DIR = PROJECT_ROOT / "config" / ("simulation" if SIMULATION_MODE else "")
STUDY_AREA_CONFIG = CONFIG_DIR / "study_area.json"
MAP_LAYERS_CONFIG = CONFIG_DIR / "map_layers.json"
PILOT_PARCELS = PROJECT_ROOT / "data" / "boundaries" / "sud_kivu_pilotes.geojson"
SAMPLE_METRICS = PROJECT_ROOT / "data" / "processed" / "parcelles_nasa_metrics.csv"
MAX_RASTER_DISPLAY_PIXELS = 1_000_000
ROLE_STYLES = {
    "boundary": {"color": "#253746", "weight": 3, "fillOpacity": 0.02},
    "pilot": {"color": "#d97706", "weight": 2, "fillColor": "#f59e0b", "fillOpacity": 0.28},
    "environmental": {"color": "#2563eb", "weight": 2, "fillColor": "#60a5fa", "fillOpacity": 0.25},
    "priority": {"color": "#b91c1c", "weight": 2, "fillColor": "#ef4444", "fillOpacity": 0.35},
    "agroforestry": {"color": "#15803d", "weight": 2, "fillColor": "#4ade80", "fillOpacity": 0.38},
    "forestry": {"color": "#166534", "weight": 2, "fillColor": "#166534", "fillOpacity": 0.38},
    "rna": {"color": "#7e22ce", "weight": 2, "fillColor": "#c084fc", "fillOpacity": 0.38},
}
ROLE_LABELS = {
    "boundary": "Limite d'étude configurée",
    "pilot": "Parcelles pilotes (démonstration)",
    "environmental": "Couche environnementale",
    "priority": "Zone prioritaire documentée",
    "agroforestry": "Zone agroforesterie documentée",
    "forestry": "Zone foresterie pure documentée",
    "rna": "Zone RNA documentée",
}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Impossible de lire la configuration {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"La configuration doit être un objet JSON : {path}")
    return value


def _project_path(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_metrics() -> dict[str, dict[str, Any]]:
    if not SAMPLE_METRICS.is_file():
        return {}
    frame = pd.read_csv(SAMPLE_METRICS)
    if "id" not in frame.columns:
        raise ValueError(f"Le CSV de métriques ne contient pas de colonne 'id': {SAMPLE_METRICS}")
    return {
        str(row["id"]): row.dropna().to_dict()
        for _, row in frame.iterrows()
        if pd.notna(row["id"])
    }


def _validate_satellite_layer(config: dict[str, Any], layer_name: str) -> None:
    satellite = config.get("satellite")
    if not isinstance(satellite, bool):
        raise ValueError(
            f"La couche « {layer_name} » doit déclarer explicitement si elle est satellitaire."
        )
    source = str(config.get("source", "")).strip()
    product = str(config.get("product", ""))
    if not source:
        raise ValueError(f"La source de la couche « {layer_name} » n'est pas renseignée.")
    if "sentinel" in f"{source} {product} {layer_name}".casefold():
        raise ValueError(
            f"La couche « {layer_name} » référence Sentinel, exclu du projet."
        )
    if satellite and "nasa" not in source.casefold():
        raise ValueError(
            f"La couche satellitaire « {layer_name} » doit déclarer une source NASA."
        )
    if satellite and not product.strip():
        raise ValueError(
            f"Le produit NASA de la couche satellitaire « {layer_name} » n'est pas renseigné."
        )


def _feature_identifier(properties: dict[str, Any], layer_name: str, index: int) -> str:
    value = properties.get("id") or properties.get("name") or properties.get("nom") or index
    return f"{layer_name}::{value}"


def _load_vector_config(config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    loaded: list[dict[str, Any]] = []
    issues: list[str] = []
    for item in config.get("layers", []):
        if not isinstance(item, dict):
            issues.append("Une entrée de couche vectorielle n'est pas un objet JSON.")
            continue
        name = str(item.get("name", "")).strip()
        role = str(item.get("role", "environmental")).strip().lower()
        path_value = item.get("path")
        if not name or not path_value:
            issues.append("Une couche vectorielle doit définir 'name' et 'path'.")
            continue
        if role not in ROLE_STYLES or role == "boundary":
            issues.append(f"Rôle non pris en charge pour « {name} »: {role}")
            continue
        try:
            _validate_satellite_layer(item, name)
        except ValueError as exc:
            issues.append(str(exc))
            continue
        path = _project_path(str(path_value))
        if not path.is_file():
            issues.append(f"Couche « {name} » introuvable : {path}")
            continue
        try:
            frame = gpd.read_file(path)
        except Exception as exc:
            issues.append(f"Lecture impossible pour « {name} » : {exc}")
            continue
        if frame.crs is None:
            issues.append(f"La couche vectorielle « {name} » n'a pas de CRS déclaré.")
            continue
        if frame.empty:
            issues.append(f"La couche vectorielle « {name} » est vide.")
            continue
        frame = frame.to_crs("EPSG:4326")
        feature_collection = json.loads(frame.to_json(drop_id=True))
        for index, feature in enumerate(feature_collection["features"]):
            properties = dict(feature.get("properties") or {})
            feature_id = _feature_identifier(properties, name, index)
            properties["_map_id"] = feature_id
            properties["_layer_name"] = name
            properties["_role"] = role
            properties["_source"] = str(item.get("source", "Non documentée"))
            properties["_product"] = str(item.get("product", ""))
            properties["_period"] = _period_label(item)
            explicit_recommendation = (
                properties.get("recommendation")
                or properties.get("recommended_intervention")
                or properties.get("intervention")
            )
            if explicit_recommendation:
                properties["_recommendation"] = str(explicit_recommendation)
            feature["properties"] = properties
        loaded.append(
            {
                "name": name,
                "role": role,
                "source": str(item.get("source", "Non documentée")),
                "period": _period_label(item),
                "geojson": feature_collection,
                "features": feature_collection["features"],
            }
        )
    return loaded, issues


def _period_label(config: dict[str, Any]) -> str:
    start = config.get("period_start")
    end = config.get("period_end")
    if start and end:
        return f"{start} – {end}"
    if start or end:
        return f"{start or 'début inconnu'} – {end or 'fin inconnue'}"
    return "Période non renseignée"


def _load_official_boundary(study_config: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    boundary_path_value = study_config.get("boundary_path")
    boundary_source = study_config.get("boundary_source")
    if not boundary_path_value:
        return None, "Aucune limite administrative officielle de Kalehe n'est configurée."
    boundary_path = _project_path(str(boundary_path_value))
    if not boundary_path.is_file():
        return None, f"Limite configurée introuvable : {boundary_path}"
    if not boundary_source:
        return None, "La source de la limite configurée n'est pas documentée."
    try:
        frame = gpd.read_file(boundary_path)
    except Exception as exc:
        return None, f"Lecture de la limite impossible : {exc}"
    if frame.crs is None or frame.empty:
        return None, "La limite doit contenir des géométries et un CRS déclaré."
    frame = frame.to_crs("EPSG:4326")
    feature_collection = json.loads(frame.to_json(drop_id=True))
    for feature in feature_collection["features"]:
        properties = dict(feature.get("properties") or {})
        properties["_map_id"] = "boundary::Kalehe"
        properties["_layer_name"] = "Territoire de Kalehe"
        properties["_role"] = "boundary"
        properties["_source"] = str(boundary_source)
        properties["_period"] = "Limite administrative"
        feature["properties"] = properties
    return {
        "name": "Territoire de Kalehe",
        "role": "boundary",
        "source": str(boundary_source),
        "period": "Limite administrative",
        "geojson": feature_collection,
        "features": feature_collection["features"],
    }, None


def _load_pilot_layer(metrics: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    if not PILOT_PARCELS.is_file():
        return None
    frame = gpd.read_file(PILOT_PARCELS)
    if frame.empty:
        return None
    if frame.crs is None:
        frame = frame.set_crs("EPSG:4326")
    else:
        frame = frame.to_crs("EPSG:4326")
    feature_collection = json.loads(frame.to_json(drop_id=True))
    for index, feature in enumerate(feature_collection["features"]):
        properties = dict(feature.get("properties") or {})
        properties.setdefault("_map_id", f"pilot::{properties.get('id', index)}")
        properties["_layer_name"] = "Parcelles pilotes (démonstration)"
        properties["_role"] = "pilot"
        properties["_source"] = "Source et méthode de numérisation non documentées"
        properties["_period"] = "Période non renseignée"
        properties["_recommendation"] = ""
        properties.update(
            {
                f"metrique_{key}": value
                for key, value in metrics.get(str(properties.get("id", "")), {}).items()
                if key not in {"id", "name"}
            }
        )
        feature["properties"] = properties
    return {
        "name": "Parcelles pilotes (démonstration)",
        "role": "pilot",
        "source": "Source et méthode de numérisation non documentées",
        "period": "Période non renseignée",
        "geojson": feature_collection,
        "features": feature_collection["features"],
    }


def _raster_overlay(
    layer: dict[str, Any],
    boundary_geojson: dict[str, Any] | None = None,
) -> tuple[np.ndarray, list[list[float]], tuple[float, float] | dict[str, str]]:
    projected, bounds = project_raster_wgs84(layer, boundary_geojson)
    return _colorize(layer, projected, bounds)


def project_raster_wgs84(
    layer: dict[str, Any],
    boundary_geojson: dict[str, Any] | None = None,
) -> tuple[np.ndarray, list[list[float]]]:
    """Clip (optional) and reproject a raster to WGS 84 for display; NaN = no data."""
    path = _project_path(str(layer["path"]))
    if not path.is_file():
        raise ValueError(f"Raster introuvable : {path}")
    with rasterio.open(path) as source:
        if source.crs is None or source.count < 1:
            raise ValueError(f"Raster sans CRS ou sans bandes : {path}")
        source_transform = source.transform
        source_width = source.width
        source_height = source.height
        if boundary_geojson:
            geometries = [
                feature["geometry"]
                for feature in boundary_geojson.get("features", [])
                if feature.get("geometry")
            ]
            if not geometries:
                raise ValueError("La limite d'étude configurée ne contient pas de géométrie.")
            boundary = unary_union([shape(geometry) for geometry in geometries])
            boundary_in_source_crs = (
                gpd.GeoSeries([boundary], crs="EPSG:4326")
                .to_crs(source.crs)
                .iloc[0]
            )
            try:
                masked, source_transform = mask_raster(
                    source,
                    [boundary_in_source_crs.__geo_interface__],
                    crop=True,
                    filled=False,
                )
            except ValueError as exc:
                raise ValueError(
                    f"Le raster « {layer.get('name', path.name)} » ne recouvre pas Kalehe."
                ) from exc
            source_data = masked[0].astype(np.float32)
            source_height, source_width = source_data.shape
        else:
            source_data = source.read(1, masked=True).astype(np.float32)
        source_values = source_data.filled(np.nan)
        source_values = source_values * float(source.scales[0]) + float(source.offsets[0])
        west, south, east, north = array_bounds(
            source_height, source_width, source_transform
        )
        transform, width, height = calculate_default_transform(
            source.crs,
            "EPSG:4326",
            source_width,
            source_height,
            west,
            south,
            east,
            north,
        )
        scale = max(
            1.0,
            math.sqrt((width * height) / MAX_RASTER_DISPLAY_PIXELS),
        )
        out_width = max(1, int(math.ceil(width / scale)))
        out_height = max(1, int(math.ceil(height / scale)))
        # ``*`` composes Affine objects in affine 2.x and 3.x; ``@`` needs affine>=3.
        transform = transform * rasterio.Affine.scale(width / out_width, height / out_height)
        projected = np.full((out_height, out_width), np.nan, dtype=np.float32)
        reproject(
            source=source_values,
            destination=projected,
            src_transform=source_transform,
            src_crs=source.crs,
            src_nodata=np.nan,
            dst_transform=transform,
            dst_crs="EPSG:4326",
            dst_nodata=np.nan,
            resampling=Resampling.nearest if layer.get("categorical", False) else Resampling.bilinear,
        )
    height, width = projected.shape
    west, south, east, north = array_bounds(height, width, transform)
    return projected, [[south, west], [north, east]]


def _colorize(
    layer: dict[str, Any],
    projected: np.ndarray,
    bounds: list[list[float]],
) -> tuple[np.ndarray, list[list[float]], tuple[float, float] | dict[str, str]]:
    path = _project_path(str(layer["path"]))
    valid = np.isfinite(projected)
    if not valid.any():
        raise ValueError(f"Raster sans pixels valides : {path}")
    rgba = np.zeros((*projected.shape, 4), dtype=np.uint8)
    if layer.get("categorical", False):
        class_colors = layer.get("class_colors")
        if not isinstance(class_colors, dict) or not class_colors:
            raise ValueError(
                f"La couche catégorielle « {layer.get('name', path.name)} » "
                "doit déclarer un dictionnaire class_colors."
            )
        for class_value, color in class_colors.items():
            try:
                rgb = _parse_hex_color(str(color))
                rgba[valid & (projected == float(class_value)), :3] = rgb
                rgba[valid & (projected == float(class_value)), 3] = 190
            except ValueError as exc:
                raise ValueError(f"Couleur/classe invalide dans « {path.name} »: {exc}") from exc
        unknown = valid & (rgba[:, :, 3] == 0)
        if unknown.any():
            raise ValueError(
                f"{int(unknown.sum())} pixels de « {path.name} » n'ont pas de couleur "
                "de classe déclarée."
            )
        class_labels = layer.get("class_labels") or {}
        legend = {
            (f"{value} ({class_labels[str(value)]})" if str(value) in class_labels else str(value)): str(color)
            for value, color in class_colors.items()
        }
    else:
        lower = float(np.nanpercentile(projected[valid], 2))
        upper = float(np.nanpercentile(projected[valid], 98))
        palette = np.array(
            [
                [239, 68, 68],
                [250, 204, 21],
                [34, 197, 94],
                [37, 99, 235],
            ],
            dtype=np.float32,
        )
        fraction = (
            np.full(projected.shape, 0.5, dtype=np.float32)
            if upper <= lower
            else np.clip((projected - lower) / (upper - lower), 0, 1)
        )
        position = np.where(valid, fraction * (len(palette) - 1), 0)
        start = np.minimum(position.astype(np.int16), len(palette) - 2)
        weight = (position - start)[..., np.newaxis]
        colors = palette[start] * (1 - weight) + palette[start + 1] * weight
        rgba[:, :, :3][valid] = colors[valid].astype(np.uint8)
        rgba[:, :, 3][valid] = 185
        legend = (lower, upper)
    return rgba, bounds, legend


def _parse_hex_color(value: str) -> tuple[int, int, int]:
    cleaned = value.lstrip("#")
    if len(cleaned) != 6:
        raise ValueError(f"Couleur hexadécimale attendue, reçu {value!r}")
    return tuple(int(cleaned[index : index + 2], 16) for index in (0, 2, 4))


def _tooltip_fields(features: list[dict[str, Any]]) -> list[str]:
    if not features:
        return ["_layer_name"]
    keys = list(features[0].get("properties", {}).keys())
    return [key for key in keys if not key.startswith("_")][:5] or ["_layer_name"]


def _add_vector_layer(map_object: folium.Map, layer: dict[str, Any], selected_id: str | None) -> None:
    role = layer["role"]
    style = ROLE_STYLES[role]
    feature_group = folium.FeatureGroup(name=layer["name"], show=True)

    def style_function(feature: dict[str, Any]) -> dict[str, Any]:
        current = dict(style)
        if feature.get("properties", {}).get("_map_id") == selected_id:
            current.update({"color": "#111827", "weight": 5, "fillOpacity": 0.55})
        return current

    folium.GeoJson(
        layer["geojson"],
        name=layer["name"],
        style_function=style_function,
        highlight_function=lambda _: {"weight": 4, "fillOpacity": 0.55},
        tooltip=folium.GeoJsonTooltip(
            fields=_tooltip_fields(layer["features"]),
            aliases=[f"{field} :" for field in _tooltip_fields(layer["features"])],
            sticky=True,
        ),
    ).add_to(feature_group)
    feature_group.add_to(map_object)


def _add_legend(
    map_object: folium.Map,
    vector_layers: list[dict[str, Any]],
    raster_legends: list[tuple[str, tuple[float, float] | dict[str, str]]],
) -> None:
    rows = []
    active_roles = {layer["role"] for layer in vector_layers}
    for role, style in ROLE_STYLES.items():
        if role not in active_roles:
            continue
        background = style.get("fillColor", "#ffffff")
        rows.append(
            f'<div><i style="background:{background};'
            f'border:1px solid {style["color"]}"></i>{ROLE_LABELS[role]}</div>'
        )
    for name, legend in raster_legends:
        if isinstance(legend, dict):
            rows.extend(
                f'<div><i style="background:{color};border:1px solid #555"></i>'
                f'{name}: classe {value}</div>'
                for value, color in legend.items()
            )
        else:
            rows.append(
                f'<div><i class="gradient"></i>{name} '
                f'({legend[0]:.3g}–{legend[1]:.3g})</div>'
            )
    html = (
        '<div style="position:fixed;bottom:28px;left:12px;z-index:9999;'
        'background:#fff;padding:10px 12px;border:1px solid #9ca3af;'
        'border-radius:6px;box-shadow:0 1px 5px #0003;font:12px/1.5 sans-serif">'
        '<b>Légende</b>'
        '<style>.kivu-legend i{display:inline-block;width:14px;height:12px;'
        'margin-right:6px;vertical-align:-2px}.gradient{background:linear-gradient'
        '(90deg,#ef4444,#facc15,#22c55e,#2563eb)}</style>'
        f'<div class="kivu-legend">{"".join(rows)}</div></div>'
    )
    map_object.get_root().html.add_child(folium.Element(html))


def _all_zone_features(layers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    zones: list[dict[str, Any]] = []
    for layer in layers:
        for feature in layer["features"]:
            geometry = feature.get("geometry")
            if geometry and geometry.get("type") in {"Polygon", "MultiPolygon"}:
                zones.append(feature)
    return zones


def _selected_raster_statistics(
    feature: dict[str, Any],
    raster_layers: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    geometry = feature.get("geometry")
    if not geometry:
        return [], ["La zone sélectionnée n'a pas de géométrie."]
    zone = gpd.GeoDataFrame(
        {"zone_id": [feature.get("properties", {}).get("_map_id", "selected")]},
        geometry=[shape(geometry)],
        crs="EPSG:4326",
    )
    statistics: list[dict[str, Any]] = []
    issues: list[str] = []
    for layer in raster_layers:
        try:
            result = extract_zonal_statistics(
                {str(layer["name"]): _project_path(str(layer["path"]))},
                zone,
                zone_id_field="zone_id",
            )
        except (GeospatialProcessingError, OSError, rasterio.errors.RasterioError) as exc:
            issues.append(f"{layer['name']}: {exc}")
            continue
        if result.empty:
            issues.append(f"{layer['name']}: aucune statistique disponible.")
            continue
        statistics.append(result.iloc[0].to_dict())
    return statistics, issues


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def render_map_page() -> None:
    """Render the interactive Folium map from configured and available project layers."""
    st.set_page_config(
        page_title="Kivu Smart — Cartographie",
        page_icon="🗺️",
        layout="wide",
    )
    st.title("Cartographie interactive — Kivu Smart")
    st.caption("Visualisation des limites, données environnementales et couches de restauration disponibles.")
    try:
        study_config = _read_json(STUDY_AREA_CONFIG)
        map_config = _read_json(MAP_LAYERS_CONFIG)
        metrics = _read_metrics()
    except (ValueError, KeyError) as exc:
        st.error(str(exc))
        return

    boundary_layer, boundary_issue = _load_official_boundary(study_config)
    vector_layers, layer_issues = _load_vector_config(map_config)
    pilot_layer = _load_pilot_layer(metrics)
    if boundary_layer:
        vector_layers.insert(0, boundary_layer)
    if pilot_layer:
        vector_layers.insert(0, pilot_layer)

    raster_layers = []
    for item in map_config.get("rasters", []):
        if not isinstance(item, dict):
            layer_issues.append("Une entrée raster n'est pas un objet JSON.")
            continue
        try:
            _validate_satellite_layer(item, str(item.get("name", "Raster")))
        except ValueError as exc:
            layer_issues.append(str(exc))
            continue
        raster_layers.append(item)

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Limite officielle", "Disponible" if boundary_layer else "Non configurée")
    col_b.metric("Entités vectorielles", sum(len(layer["features"]) for layer in vector_layers))
    col_c.metric("Couches raster", len(raster_layers))

    if boundary_issue:
        st.warning(
            f"{boundary_issue} Les parcelles pilotes éventuelles ne représentent pas "
            "la limite administrative de Kalehe."
        )
    if layer_issues:
        with st.expander("Couches non affichées — vérifier les données", expanded=True):
            for issue in layer_issues:
                st.warning(issue)
    if not raster_layers and not vector_layers:
        st.info("Aucune couche cartographique exploitable n'est configurée.")
        return

    zones = _all_zone_features(vector_layers)
    zone_options = {
        str(feature["properties"].get("_map_id")): (
            f'{feature["properties"].get("_layer_name", "Zone")} — '
            f'{feature["properties"].get("name") or feature["properties"].get("nom") or feature["properties"].get("id") or feature["properties"].get("_map_id")}'
        )
        for feature in zones
    }
    selected_id: str | None = None
    selected_feature: dict[str, Any] | None = None
    if zone_options:
        selected_id = st.selectbox(
            "Sélectionner une zone",
            options=list(zone_options.keys()),
            format_func=lambda value: zone_options[value],
        )
        selected_feature = next(
            feature
            for feature in zones
            if feature["properties"].get("_map_id") == selected_id
        )
    else:
        st.info("Aucune zone polygonale sélectionnable n'est disponible.")

    map_bounds: list[list[float]] = [
        point
        for layer in vector_layers
        for feature in layer["features"]
        for point in _geometry_bounds_wgs84(feature.get("geometry"))
    ]
    initial_location = (
        [
            (min(point[0] for point in map_bounds) + max(point[0] for point in map_bounds)) / 2,
            (min(point[1] for point in map_bounds) + max(point[1] for point in map_bounds)) / 2,
        ]
        if map_bounds
        else [-2.3, 28.8]
    )
    map_object = folium.Map(
        location=initial_location,
        zoom_start=9,
        tiles="OpenStreetMap",
        control_scale=True,
        prefer_canvas=True,
    )
    raster_legends: list[tuple[str, tuple[float, float] | dict[str, str]]] = []
    visible_raster_layers: list[dict[str, Any]] = []

    for layer in vector_layers:
        try:
            _add_vector_layer(map_object, layer, selected_id)
        except Exception as exc:
            st.error(f"Erreur d'affichage de « {layer['name']} » : {exc}")

    for item in raster_layers:
        name = str(item.get("name", "Raster"))
        path_value = item.get("path")
        if not name or not path_value:
            st.error("Une couche raster doit définir un nom et un chemin.")
            continue
        path = _project_path(str(path_value))
        if not path.is_file():
            st.warning(f"Raster « {name} » introuvable : {path}")
            continue
        try:
            display_data, bounds, legend = _raster_overlay(
                item,
                boundary_layer["geojson"] if boundary_layer else None,
            )
            ImageOverlay(
                image=display_data,
                bounds=bounds,
                name=name,
                opacity=0.8,
                show=bool(item.get("show", True)),
                interactive=False,
                cross_origin=False,
                zindex=1,
            ).add_to(map_object)
            raster_legends.append((name, legend))
            visible_raster_layers.append(item)
            map_bounds.extend(bounds)
        except Exception as exc:
            st.error(f"Erreur de préparation du raster « {name} » : {exc}")

    _add_legend(map_object, vector_layers, raster_legends)
    folium.LayerControl(collapsed=False).add_to(map_object)
    if map_bounds:
        south = min(point[0] for point in map_bounds)
        west = min(point[1] for point in map_bounds)
        north = max(point[0] for point in map_bounds)
        east = max(point[1] for point in map_bounds)
        map_object.fit_bounds([[south, west], [north, east]], padding=(20, 20))

    st_folium(
        map_object,
        height=680,
        use_container_width=True,
        returned_objects=[],
        key="kivu_interactive_map",
    )

    active_roles = {layer["role"] for layer in vector_layers}
    missing_restoration_roles = {
        "priority": "zones prioritaires",
        "agroforestry": "zones d'agroforesterie",
        "forestry": "zones de foresterie pure",
        "rna": "zones de régénération naturelle assistée",
    }
    absent_roles = [
        label
        for role, label in missing_restoration_roles.items()
        if role not in active_roles
    ]
    if absent_roles:
        st.info(
            "Couches de restauration non configurées : "
            + ", ".join(absent_roles)
            + ". Aucune zone n'est inférée par la carte."
        )

    if selected_feature is not None:
        properties = selected_feature.get("properties", {})
        st.subheader("Caractéristiques de la zone sélectionnée")
        public_properties = {
            key: _format_value(value)
            for key, value in properties.items()
            if not key.startswith("_") and value is not None
        }
        if public_properties:
            st.dataframe(
                pd.DataFrame(
                    [{"Caractéristique": key, "Valeur": value} for key, value in public_properties.items()]
                ),
                hide_index=True,
                width="stretch",
            )
        else:
            st.info("Aucune caractéristique attributaire n'est disponible pour cette zone.")

        recommendation = properties.get("_recommendation")
        if recommendation:
            st.markdown(f"**Intervention documentée :** {recommendation}")
            justification = properties.get("justification") or properties.get("rationale")
            if justification:
                st.write(f"**Justification :** {justification}")
        else:
            st.info(
                "Aucune recommandation validée n'est associée à cette zone. "
                "La carte n'en déduit pas à partir de données de démonstration."
            )

        if properties.get("_role") == "pilot":
            sample_metrics = {
                key.removeprefix("metrique_"): _format_value(value)
                for key, value in properties.items()
                if key.startswith("metrique_")
            }
            if sample_metrics:
                st.warning(
                    "Les valeurs ci-dessous proviennent du CSV de démonstration; "
                    "leur source et leur validation ne sont pas établies. "
                    "Elles ne constituent pas une recommandation."
                )
                st.dataframe(
                    pd.DataFrame(
                        [{"Mesure non vérifiée": key, "Valeur": value} for key, value in sample_metrics.items()]
                    ),
                    hide_index=True,
                    width="stretch",
                )
        raster_statistics, raster_issues = _selected_raster_statistics(
            selected_feature, visible_raster_layers
        )
        if raster_statistics:
            st.markdown("**Statistiques des rasters environnementaux sur cette zone**")
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Couche": row["layer"],
                            "Pixels valides": row["valid_pixel_count"],
                            "Moyenne": _format_value(row["mean"]),
                            "Écart-type": _format_value(row["std"]),
                            "Minimum": _format_value(row["min"]),
                            "Maximum": _format_value(row["max"]),
                            "Unité": row["unit"] or "Non renseignée",
                            "Résolution native X": _format_value(row["resolution_x"]),
                            "Résolution native Y": _format_value(row["resolution_y"]),
                        }
                        for row in raster_statistics
                    ]
                ),
                hide_index=True,
                width="stretch",
            )
        for issue in raster_issues:
            st.warning(f"Statistique raster indisponible : {issue}")
        st.caption(
            f"Couche : {properties.get('_layer_name', 'inconnue')} · "
            f"Source : {properties.get('_source', 'non documentée')} · "
            f"Période : {properties.get('_period', 'non renseignée')}"
        )

    st.caption(
        "Fond OpenStreetMap utilisé comme fond de référence. Les fonds et couches "
        "satellitaires ne sont affichés que s'ils sont déclarés dans la configuration "
        "avec une provenance NASA vérifiable; les sources Sentinel sont exclues."
    )


def _geometry_bounds_wgs84(geometry: dict[str, Any] | None) -> list[list[float]]:
    if not geometry:
        return []
    try:
        from shapely.geometry import shape

        min_x, min_y, max_x, max_y = shape(geometry).bounds
        return [[min_y, min_x], [max_y, max_x]]
    except Exception:
        return []
