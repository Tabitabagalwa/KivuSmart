"""Construction de la carte interactive (Folium / Leaflet)."""

from __future__ import annotations

import json
import math
from html import escape
from typing import Any

import folium
import geopandas as gpd
import numpy as np
from folium import plugins
from folium.raster_layers import ImageOverlay
from shapely.geometry import mapping, shape
from shapely.ops import unary_union

from .donnees import Donnees, raster_projete
from .theme import ORDRE, PALETTES_RASTER, STRATEGIES, nombre

FONDS = {
    "Imagerie satellite": {
        "tiles": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        "attr": "Imagerie © Esri, Maxar, Earthstar Geographics",
        "max_zoom": 19,
    },
    "Plan clair": {
        "tiles": "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
        "attr": "© OpenStreetMap contributors © CARTO",
        "max_zoom": 19,
    },
    "Topographie": {
        "tiles": "https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
        "attr": "© OpenStreetMap contributors, SRTM | Style © OpenTopoMap (CC-BY-SA)",
        "max_zoom": 17,
    },
    "OpenStreetMap": {
        "tiles": "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
        "attr": "© OpenStreetMap contributors",
        "max_zoom": 19,
    },
    "NASA GIBS — MODIS vraies couleurs": {
        "tiles": ("https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/MODIS_Terra_CorrectedReflectance_TrueColor/"
                  "default/2025-07-15/GoogleMapsCompatible_Level9/{z}/{y}/{x}.jpg"),
        "attr": "Imagerie NASA EOSDIS GIBS (MODIS Terra, 15/07/2025)",
        "max_zoom": 18,
        "max_native_zoom": 9,
    },
}

STYLE_CONTEXTE = {
    "parc": {"color": "#166534", "weight": 2, "dashArray": "6 4", "fillColor": "#22c55e", "fillOpacity": 0.10},
    "eau": {"color": "#1d4ed8", "weight": 1, "fillColor": "#3b82f6", "fillOpacity": 0.35},
    "riviere": {"color": "#38bdf8", "weight": 2.2, "opacity": 0.95},
    "route": {"color": "#57534e", "weight": 2, "opacity": 0.85},
    "autre": {"color": "#2563eb", "weight": 1.5, "fillOpacity": 0.15},
}


def type_contexte(couche: dict[str, Any]) -> str:
    nom = couche["name"].casefold()
    if any(m in nom for m in ("parc", "aire", "pnkb", "protég")):
        return "parc"
    if any(m in nom for m in ("lac", "plan d'eau")):
        return "eau"
    if any(m in nom for m in ("rivi", "cours d", "fleuve")):
        return "riviere"
    if "route" in nom or "piste" in nom:
        return "route"
    if any(m in nom for m in ("localit", "village", "habitat")):
        return "localite"
    return "autre"


# --------------------------------------------------------------------------- rasters
def coloriser(couche: dict[str, Any], valeurs: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Image RGBA et légende d'un raster projeté (NaN = transparent)."""
    valide = np.isfinite(valeurs)
    rgba = np.zeros((*valeurs.shape, 4), dtype=np.uint8)
    if couche.get("categorical"):
        couleurs = couche.get("class_colors") or {}
        noms = couche.get("class_labels") or {}
        for valeur, couleur in couleurs.items():
            masque = valide & (valeurs == float(valeur))
            rgba[masque, :3] = [int(couleur.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4)]
            rgba[masque, 3] = 205
        return rgba, {"type": "classes", "classes": [(noms.get(str(v), f"Classe {v}"), c) for v, c in couleurs.items()]}

    cle = couche.get("palette", "vegetation")
    palette = np.array([[int(c.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4)]
                        for c in PALETTES_RASTER.get(cle, PALETTES_RASTER["vegetation"])], dtype=np.float32)
    donnees = valeurs[valide]
    if cle == "variation":  # divergente centrée sur zéro
        borne = float(np.nanpercentile(np.abs(donnees), 98)) or 1.0
        bas, haut = -borne, borne
    else:
        bas, haut = float(np.nanpercentile(donnees, 2)), float(np.nanpercentile(donnees, 98))
    fraction = np.clip((valeurs - bas) / (haut - bas), 0, 1) if haut > bas else np.full(valeurs.shape, 0.5)
    position = np.where(valide, fraction * (len(palette) - 1), 0)
    debut = np.minimum(position.astype(int), len(palette) - 2)
    poids = (position - debut)[..., None]
    rgba[..., :3] = np.where(valide[..., None], palette[debut] * (1 - poids) + palette[debut + 1] * poids, 0)
    rgba[..., 3] = np.where(valide, 215, 0)
    return rgba, {"type": "gradient", "min": bas, "max": haut, "unite": couche.get("unit", ""),
                  "couleurs": PALETTES_RASTER.get(cle, PALETTES_RASTER["vegetation"])}


# ----------------------------------------------------------------------------- fiches
def _ligne(libelle: str, valeur: str) -> str:
    return (f'<tr><td style="color:#6b7280;padding:3px 10px 3px 0">{escape(libelle)}</td>'
            f'<td style="color:#111827;font-weight:600;padding:3px 0">{escape(valeur)}</td></tr>')


def fiche_html(zone: dict[str, Any]) -> str:
    strategie = STRATEGIES[str(zone["strategie"])]
    lignes = [_ligne("Surface", f"{nombre(zone['surface_ha'], 1)} ha")]
    if _ok(zone.get("aptitude")):
        lignes.append(_ligne("Aptitude", f"{nombre(zone['aptitude'], 1)} / 5 ({zone.get('priorite') or '—'})"))
    for cle, libelle, fmt in (("pente", "Pente moyenne", "{} %"), ("ndvi", "NDVI 2025", "{}"),
                              ("variation_ndvi", "Variation NDVI 2015-2025", "{}"),
                              ("altitude", "Altitude moyenne", "{} m")):
        if _ok(zone.get(cle)):
            valeur = zone[cle]
            texte = nombre(valeur, 0 if cle in ("pente", "altitude") else 2)
            if cle == "variation_ndvi" and valeur > 0:
                texte = "+" + texte
            lignes.append(_ligne(libelle, fmt.format(texte)))
    justification = zone.get("justification") or ""
    return (
        '<div style="font-family:system-ui,Segoe UI,sans-serif;min-width:270px;max-width:330px">'
        f'<div style="border-left:5px solid {strategie.couleur};padding:2px 0 2px 9px;margin-bottom:6px">'
        f'<div style="font-size:11px;color:#6b7280;text-transform:uppercase;letter-spacing:.04em">'
        f'{escape(strategie.nom)}</div>'
        f'<div style="font-size:14px;font-weight:700;color:#111827">{escape(str(zone["nom"]))}</div></div>'
        f'<table style="font-size:12px;border-collapse:collapse;width:100%">{"".join(lignes)}</table>'
        + (f'<div style="font-size:12px;color:#374151;margin-top:7px;line-height:1.4">'
           f'{escape(str(justification))}</div>' if justification else "")
        + "</div>"
    )


def _ok(valeur: Any) -> bool:
    return valeur is not None and not (isinstance(valeur, float) and math.isnan(valeur))


# ----------------------------------------------------------------------------- légende
def _legende(zones: gpd.GeoDataFrame, contexte: list[str], rasters: list[tuple[str, dict]]) -> str:
    blocs = ['<div class="kv-leg-titre">Interventions</div>']
    for cle in ORDRE:
        sous = zones[zones["strategie"] == cle]
        if sous.empty:
            continue
        s = STRATEGIES[cle]
        bord = "border:2px dashed #7f1d1d;" if cle == "prioritaire" else f"border:1px solid {s.couleur};"
        blocs.append(f'<div class="kv-leg-ligne"><i style="background:{s.couleur};{bord}"></i>'
                     f'{escape(s.nom)} <span class="kv-leg-n">{len(sous)} · {nombre(sous["surface_ha"].sum())} ha</span></div>')
    symboles = {"parc": ("#22c55e33", "2px dashed #166534"), "eau": ("#3b82f6aa", "1px solid #1d4ed8"),
                "riviere": ("#38bdf8", "0"), "route": ("#57534e", "0"),
                "localite": ("#ffffff", "2px solid #111827")}
    if contexte:
        blocs.append('<div class="kv-leg-titre">Repères</div>')
        for nom, genre in contexte:
            fond, bord = symboles.get(genre, ("#2563eb33", "1px solid #2563eb"))
            forme = "border-radius:50%;" if genre == "localite" else ("height:4px;margin-top:5px;" if genre in ("riviere", "route") else "")
            blocs.append(f'<div class="kv-leg-ligne"><i style="background:{fond};border:{bord};{forme}"></i>{escape(nom)}</div>')
    for nom, leg in rasters:
        blocs.append(f'<div class="kv-leg-titre">{escape(nom)}</div>')
        if leg["type"] == "classes":
            blocs += [f'<div class="kv-leg-ligne"><i style="background:{c}"></i>{escape(n)}</div>' for n, c in leg["classes"]]
        else:
            degrade = ",".join(leg["couleurs"])
            unite = f' {escape(leg["unite"])}' if leg["unite"] and leg["unite"] != "sans unité" else ""
            blocs.append(f'<div class="kv-leg-degrade" style="background:linear-gradient(90deg,{degrade})"></div>'
                         f'<div class="kv-leg-bornes"><span>{leg["min"]:.3g}{unite}</span><span>{leg["max"]:.3g}{unite}</span></div>')
    return (
        "<style>"
        ".kv-leg{position:absolute;bottom:34px;left:10px;z-index:9999;background:rgba(255,255,255,.96);"
        "padding:10px 12px;border-radius:10px;box-shadow:0 2px 10px rgba(0,0,0,.18);"
        "font:12px/1.45 system-ui,Segoe UI,sans-serif;color:#111827;max-height:58%;overflow:auto;min-width:210px}"
        ".kv-leg-titre{font-weight:700;margin:6px 0 3px;font-size:11px;text-transform:uppercase;"
        "letter-spacing:.05em;color:#4b5563}.kv-leg-titre:first-child{margin-top:0}"
        ".kv-leg-ligne{display:flex;align-items:center;gap:7px;margin:2px 0}"
        ".kv-leg-ligne i{display:inline-block;width:16px;height:11px;border-radius:2px;flex:none}"
        ".kv-leg-n{color:#6b7280;margin-left:auto;padding-left:8px;white-space:nowrap}"
        ".kv-leg-degrade{height:9px;border-radius:3px;margin-top:3px}"
        ".kv-leg-bornes{display:flex;justify-content:space-between;color:#4b5563;font-size:11px}"
        ".kv-nord{position:absolute;top:12px;left:56px;z-index:9999;background:rgba(255,255,255,.96);"
        "border-radius:8px;box-shadow:0 1px 6px rgba(0,0,0,.2);padding:3px 8px;text-align:center;"
        "font:700 12px system-ui,sans-serif;color:#111827;line-height:1.05}"
        "</style>"
        f'<div class="kv-leg">{"".join(blocs)}</div>'
        '<div class="kv-nord" title="Nord"><div style="font-size:15px">▲</div>N</div>'
    )


# ------------------------------------------------------------------------------- carte
def construire_carte(
    donnees: Donnees,
    zones: gpd.GeoDataFrame,
    *,
    fond: str = "Imagerie satellite",
    rasters: list[dict[str, Any]] | None = None,
    opacite: float = 0.75,
    contexte_visible: set[str] | None = None,
    selection: str | None = None,
    outils: bool = True,
) -> folium.Map:
    carte = folium.Map(location=[-2.32, 28.81], zoom_start=12, tiles=None, control_scale=True,
                       prefer_canvas=False, zoom_control=True)
    for nom, options in FONDS.items():
        folium.TileLayer(
            tiles=options["tiles"], attr=options["attr"], name=nom, overlay=False, control=True,
            show=(nom == fond), max_zoom=options["max_zoom"],
            max_native_zoom=options.get("max_native_zoom", options["max_zoom"]),
        ).add_to(carte)

    limite_geojson = donnees.limite["geojson"] if donnees.limite else None
    legendes_rasters = []
    for couche in rasters or []:
        valeurs, bornes = raster_projete(json.dumps(couche, sort_keys=True),
                                         json.dumps(limite_geojson) if limite_geojson else "")
        image, legende = coloriser(couche, valeurs)
        ImageOverlay(image=image, bounds=bornes, name=f"▦ {couche['name']}", opacity=opacite,
                     interactive=False, cross_origin=False, zindex=2, mercator_project=True).add_to(carte)
        legendes_rasters.append((couche["name"], legende))

    contexte_legende = []
    for couche in donnees.contexte:
        if contexte_visible is not None and couche["name"] not in contexte_visible:
            continue
        genre = type_contexte(couche)
        groupe = folium.FeatureGroup(name=couche["name"], show=True)
        if genre == "localite":
            for feature in couche["features"]:
                centre = shape(feature["geometry"]).centroid
                nom = feature["properties"].get("nom") or feature["properties"].get("name") or "Localité"
                folium.CircleMarker([centre.y, centre.x], radius=5, color="#111827", weight=2,
                                    fill=True, fill_color="#ffffff", fill_opacity=1,
                                    tooltip=str(nom)).add_to(groupe)
        else:
            style = STYLE_CONTEXTE.get(genre, STYLE_CONTEXTE["autre"])
            folium.GeoJson(
                couche["geojson"], style_function=lambda _f, s=style: s,
                tooltip=folium.GeoJsonTooltip(fields=["nom"], labels=False) if _a_champ(couche, "nom") else None,
            ).add_to(groupe)
        groupe.add_to(carte)
        contexte_legende.append((couche["name"], genre))

    if limite_geojson:
        for couleur, epaisseur in (("#ffffff", 5), ("#0f172a", 2)):
            folium.GeoJson(limite_geojson, name="Limite de la zone d'étude" if couleur == "#0f172a" else None,
                           control=couleur == "#0f172a",
                           style_function=lambda _f, c=couleur, e=epaisseur: {
                               "color": c, "weight": e, "fill": False, "dashArray": "8 5" if c == "#0f172a" else None},
                           ).add_to(carte)

    for cle in ORDRE:
        sous = zones[zones["strategie"] == cle]
        if sous.empty:
            continue
        s = STRATEGIES[cle]
        donnees_zone = sous.drop(columns=["strategie"]).assign(strategie=cle)
        donnees_zone["fiche"] = [fiche_html(z) for z in donnees_zone.to_dict("records")]
        donnees_zone["aptitude_txt"] = [f"{a:.1f}/5" if _ok(a) else "—" for a in donnees_zone["aptitude"]]
        geojson = json.loads(donnees_zone[["map_id", "nom", "aptitude_txt", "fiche", "geometry"]].to_json())
        style = {"color": s.couleur, "weight": 1.4, "fillColor": s.couleur, "fillOpacity": 0.55}
        if cle == "prioritaire":
            style = {"color": "#7f1d1d", "weight": 2.2, "dashArray": "5 4", "fillColor": s.couleur, "fillOpacity": 0.45}
        groupe = folium.FeatureGroup(
            name=f'<span style="color:{s.couleur}">■</span> {s.nom} ({len(sous)})', show=True)
        folium.GeoJson(
            geojson,
            style_function=lambda _f, st_=style: st_,
            highlight_function=lambda _f: {"weight": 3, "fillOpacity": 0.78},
            tooltip=folium.GeoJsonTooltip(fields=["nom", "aptitude_txt"], aliases=["", "Aptitude"],
                                          labels=True, sticky=True, localize=False),
            popup=folium.GeoJsonPopup(fields=["fiche"], labels=False, localize=False, max_width=360),
        ).add_to(groupe)
        groupe.add_to(carte)

    cible = None
    if selection is not None and not zones.empty and selection in set(zones["map_id"]):
        geometrie = zones.loc[zones["map_id"] == selection, "geometry"].iloc[0]
        for couleur, epaisseur in (("#ffffff", 7), ("#facc15", 3.5)):
            folium.GeoJson(mapping(geometrie), control=False,
                           style_function=lambda _f, c=couleur, e=epaisseur: {"color": c, "weight": e, "fill": False},
                           ).add_to(carte)
        cible = geometrie.bounds

    if outils:
        plugins.Fullscreen(position="topleft", title="Plein écran", title_cancel="Quitter le plein écran").add_to(carte)
        plugins.MeasureControl(position="topleft", primary_length_unit="kilometers",
                               secondary_length_unit="meters", primary_area_unit="hectares",
                               secondary_area_unit="sqmeters").add_to(carte)
        plugins.MousePosition(position="bottomright", separator=" · ", prefix="Lat/Lon", num_digits=5).add_to(carte)
        plugins.MiniMap(tile_layer=folium.TileLayer(FONDS["Plan clair"]["tiles"], attr=FONDS["Plan clair"]["attr"]),
                        toggle_display=True, position="bottomright", width=140, height=140,
                        zoom_level_offset=-5, minimized=True).add_to(carte)
        folium.LayerControl(position="topright", collapsed=True).add_to(carte)
        carte.get_root().html.add_child(folium.Element(_legende(zones, contexte_legende, legendes_rasters)))

    if cible:
        ouest, sud, est, nord = cible
        carte.fit_bounds([[sud, ouest], [nord, est]], padding=(60, 60), max_zoom=15)
    elif limite_geojson:
        geometrie = unary_union([shape(f["geometry"]) for f in limite_geojson["features"]])
        ouest, sud, est, nord = geometrie.bounds
        carte.fit_bounds([[sud, ouest], [nord, est]], padding=(10, 10))
    elif not zones.empty:
        ouest, sud, est, nord = zones.total_bounds
        carte.fit_bounds([[sud, ouest], [nord, est]], padding=(10, 10))
    return carte


def _a_champ(couche: dict[str, Any], champ: str) -> bool:
    return all(champ in (f.get("properties") or {}) for f in couche["features"])
