"""Chargement des données affichées par l'interface (avec mise en cache Streamlit).

Choix du jeu de données (variable d'environnement ``KIVU_DONNEES``) :

* ``reel``  : ``config/`` (données NASA et couches locales réelles) ;
* ``demo``  : ``config/simulation/`` (jeu de démonstration synthétique) ;
* absente   : ``config/`` dès qu'il contient des couches, sinon le jeu de démonstration.

L'ancienne variable ``KIVU_SIMULATION=1`` force aussi le jeu de démonstration.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import streamlit as st
from shapely.geometry import shape

from . import map_view as mv
from .theme import ORDRE

RACINE = Path(__file__).resolve().parents[1]
CONFIG_REEL = RACINE / "config"
CONFIG_DEMO = RACINE / "config" / "simulation"
REFERENTIEL = RACINE / "data" / "reference" / "referentiel_kivu.json"
CRS_METRIQUE = "EPSG:32735"

ROLE_VERS_STRATEGIE = {"rna": "rna", "agroforestry": "agroforesterie", "forestry": "reboisement",
                       "priority": "prioritaire"}
COLONNES_ZONES = {
    "aptitude_sur_5": "aptitude", "priorite": "priorite", "surface_ha": "surface_ha",
    "pente_moyenne_pct": "pente", "ndvi_2025_moyen": "ndvi", "variation_ndvi_2015_2025": "variation_ndvi",
    "altitude_moyenne_m": "altitude", "justification": "justification",
}


@dataclass
class Donnees:
    dossier: str
    demo: bool
    nom_zone: str
    source_limite: str
    limite: dict[str, Any] | None
    surface_zone_ha: float | None
    zones: gpd.GeoDataFrame
    contexte: list[dict[str, Any]]
    rasters: list[dict[str, Any]]
    problemes: list[str] = field(default_factory=list)


def dossier_config() -> Path:
    mode = os.getenv("KIVU_DONNEES", "").strip().lower()
    if mode in {"demo", "demonstration", "simulation"} or \
            os.getenv("KIVU_SIMULATION", "").strip().lower() in {"1", "true", "oui", "yes"}:
        return CONFIG_DEMO
    if mode in {"reel", "réel", "real"}:
        return CONFIG_REEL
    try:
        couches = json.loads((CONFIG_REEL / "map_layers.json").read_text(encoding="utf-8"))
        a_des_couches = bool(couches.get("layers") or couches.get("rasters"))
    except (OSError, json.JSONDecodeError):
        a_des_couches = False
    return CONFIG_REEL if a_des_couches or not CONFIG_DEMO.is_dir() else CONFIG_DEMO


def _signature(dossier: Path) -> float:
    fichiers = [dossier / "study_area.json", dossier / "map_layers.json"]
    try:
        config = json.loads((dossier / "map_layers.json").read_text(encoding="utf-8"))
        fichiers += [mv._project_path(str(i["path"])) for i in config.get("layers", []) + config.get("rasters", [])
                     if isinstance(i, dict) and i.get("path")]
    except (OSError, json.JSONDecodeError):
        pass
    return sum(f.stat().st_mtime for f in fichiers if f.is_file())


def charger() -> Donnees:
    dossier = dossier_config()
    return _charger(str(dossier), _signature(dossier))


def _surface_ha(geometries) -> float:
    serie = gpd.GeoSeries(geometries, crs="EPSG:4326").to_crs(CRS_METRIQUE)
    return float(serie.area.sum() / 10_000)


@st.cache_data(show_spinner="Chargement des données…")
def _charger(dossier: str, _signature_fichiers: float) -> Donnees:
    chemin = Path(dossier)
    problemes: list[str] = []
    etude = mv._read_json(chemin / "study_area.json")
    config = mv._read_json(chemin / "map_layers.json")
    limite, probleme_limite = mv._load_official_boundary(etude)
    if probleme_limite:
        problemes.append(probleme_limite)
    couches, erreurs = mv._load_vector_config(config)
    problemes += erreurs

    lignes, contexte = [], []
    for couche in couches:
        strategie = ROLE_VERS_STRATEGIE.get(couche["role"])
        if strategie is None:
            contexte.append(couche)
            continue
        for feature in couche["features"]:
            p = feature.get("properties", {})
            geometrie = shape(feature["geometry"])
            ligne = {
                "map_id": p.get("_map_id"),
                "id": str(p.get("id") or p.get("_map_id")),
                "nom": str(p.get("nom") or p.get("name") or p.get("id") or "Zone"),
                "strategie": strategie,
                "intervention": str(p.get("_recommendation") or couche["name"]),
                "source": p.get("_source", ""),
                "geometry": geometrie,
            }
            for origine, cible in COLONNES_ZONES.items():
                ligne[cible] = p.get(origine)
            lignes.append(ligne)

    colonnes = ["map_id", "id", "nom", "strategie", "intervention", "source", *COLONNES_ZONES.values(), "geometry"]
    zones = gpd.GeoDataFrame(lignes or None, columns=colonnes, geometry="geometry", crs="EPSG:4326")
    for colonne in ("aptitude", "surface_ha", "pente", "ndvi", "variation_ndvi", "altitude"):
        zones[colonne] = pd.to_numeric(zones[colonne], errors="coerce")
    if not zones.empty and zones["surface_ha"].isna().any():
        manquantes = zones["surface_ha"].isna()
        zones.loc[manquantes, "surface_ha"] = (
            zones.loc[manquantes].to_crs(CRS_METRIQUE).area / 10_000).round(1)
    zones["strategie"] = pd.Categorical(zones["strategie"], categories=list(ORDRE), ordered=True)
    zones = zones.sort_values(["strategie", "aptitude"], ascending=[True, False]).reset_index(drop=True)

    rasters = []
    for item in config.get("rasters", []):
        if not isinstance(item, dict):
            continue
        try:
            mv._validate_satellite_layer(item, str(item.get("name", "Raster")))
        except ValueError as exc:
            problemes.append(str(exc))
            continue
        if mv._project_path(str(item.get("path", ""))).is_file():
            rasters.append(item)
        else:
            problemes.append(f"Raster introuvable : {item.get('name')}")

    surface = None
    if limite:
        surface = _surface_ha([shape(f["geometry"]) for f in limite["features"]])
    return Donnees(
        dossier=dossier,
        demo=chemin.resolve() == CONFIG_DEMO.resolve(),
        nom_zone=str(etude.get("name") or "Zone d'étude"),
        source_limite=str(etude.get("boundary_source") or "Non documentée"),
        limite=limite,
        surface_zone_ha=surface,
        zones=zones,
        contexte=contexte,
        rasters=rasters,
        problemes=problemes,
    )


@st.cache_data(show_spinner=False)
def raster_projete(couche_json: str, limite_json: str) -> tuple[np.ndarray, list[list[float]]]:
    couche = json.loads(couche_json)
    limite = json.loads(limite_json) if limite_json else None
    return mv.project_raster_wgs84(couche, limite)


CLIMAT = RACINE / "data" / "processed" / "climat"
POWER_MENSUEL = RACINE / "data" / "raw" / "power" / "power_mensuel_2006_2025.csv"
RAPPORT_REEL = RACINE / "data" / "reel" / "rapport_qualite.json"


@dataclass
class Climat:
    annuel: pd.DataFrame
    mensuel: pd.DataFrame
    anomalies: pd.DataFrame
    synthese: pd.DataFrame
    meta: dict[str, Any]
    power: pd.DataFrame | None


def climat() -> Climat | None:
    """Projections CMIP6 résumées (``scripts/telecharger_cmip6.py``), ou ``None`` si absentes."""
    fichiers = [CLIMAT / f"cmip6_{n}" for n in ("annuel.csv", "mensuel.csv", "anomalies.csv", "synthese.csv",
                                                 "meta.json")]
    if not all(f.is_file() for f in fichiers):
        return None
    return _climat(sum(f.stat().st_mtime for f in fichiers))


@st.cache_data(show_spinner="Chargement des projections climatiques…")
def _climat(_signature_fichiers: float) -> Climat:
    power = None
    if POWER_MENSUEL.is_file():
        brut = pd.read_csv(POWER_MENSUEL, dtype={"periode": str})
        brut = brut[brut["periode"].str.len() == 6]
        brut["annee"] = brut["periode"].str[:4].astype(int)
        brut["mois"] = brut["periode"].str[4:].astype(int)
        brut = brut[brut["mois"].between(1, 12)]
        jours = pd.to_datetime(dict(year=brut["annee"], month=brut["mois"], day=1)).dt.days_in_month
        brut["pluie_mm"] = brut["PRECTOTCORR"] * jours
        power = brut
    return Climat(
        annuel=pd.read_csv(CLIMAT / "cmip6_annuel.csv"),
        mensuel=pd.read_csv(CLIMAT / "cmip6_mensuel.csv"),
        anomalies=pd.read_csv(CLIMAT / "cmip6_anomalies.csv"),
        synthese=pd.read_csv(CLIMAT / "cmip6_synthese.csv"),
        meta=json.loads((CLIMAT / "cmip6_meta.json").read_text(encoding="utf-8")),
        power=power,
    )


def rapport_reel() -> dict[str, Any]:
    try:
        return json.loads(RAPPORT_REEL.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


@st.cache_data(show_spinner=False)
def referentiel() -> dict[str, Any]:
    try:
        return json.loads(REFERENTIEL.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def synthese(zones: gpd.GeoDataFrame) -> pd.DataFrame:
    """Nombre de zones, surface et aptitude moyenne par stratégie (ordre fixe)."""
    if zones.empty:
        return pd.DataFrame(columns=["strategie", "zones", "surface_ha", "aptitude"])
    tableau = (zones.groupby("strategie", observed=False)
               .agg(zones=("id", "count"), surface_ha=("surface_ha", "sum"), aptitude=("aptitude", "mean"))
               .reset_index())
    tableau["strategie"] = tableau["strategie"].astype(str)
    return tableau
