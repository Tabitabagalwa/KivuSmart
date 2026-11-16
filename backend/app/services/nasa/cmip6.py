"""Projections climatiques CMIP6 réduites d'échelle par la NASA (NEX-GDDP-CMIP6).

Jeu de données : NASA Earth Exchange Global Daily Downscaled Projections (NEX-GDDP-CMIP6),
grille de 0,25° (~27 km), valeurs journalières 1950-2100 corrigées des biais (BCSD) par
rapport à une climatologie observée. Accès sans authentification via le serveur THREDDS
du NASA Center for Climate Simulation (``ds.nccs.nasa.gov``) :

* ``catalog/.../catalog.xml``  liste les fichiers annuels d'une variable ;
* ``ncss/grid/...``            découpe un fichier sur une emprise et renvoie du NetCDF-3,
  lisible avec :mod:`scipy.io` (aucune dépendance NetCDF supplémentaire).

Une cellule de 27 km mélange rive du lac et hautes terres : les valeurs décrivent une
tendance régionale, pas le climat d'une parcelle.
"""

from __future__ import annotations

import io
import math
import re
from dataclasses import dataclass

import numpy as np

from .client import NasaHttpClient
from .errors import NasaApiError

SERVICE = "nex-gddp-cmip6"
THREDDS = "https://ds.nccs.nasa.gov/thredds"
RACINE_DONNEES = "AMES/NEX/GDDP-CMIP6"
SOURCE = "NASA NEX-GDDP-CMIP6 (NCCS THREDDS, ds.nccs.nasa.gov)"
DOI = "https://doi.org/10.7917/OFSG3345"

SCENARIOS = {
    "historical": "Historique",
    "ssp126": "SSP1-2.6 (forte atténuation)",
    "ssp245": "SSP2-4.5 (intermédiaire)",
    "ssp370": "SSP3-7.0 (rivalités régionales)",
    "ssp585": "SSP5-8.5 (fortes émissions)",
}
VARIABLES = {
    "pr": "Précipitations (mm/jour)",
    "tas": "Température moyenne de l'air (°C)",
    "tasmax": "Température maximale (°C)",
    "tasmin": "Température minimale (°C)",
}
# Modèles retenus : membre r1i1p1f1 disponible pour tous les scénarios, familles de
# modèles différentes (pour couvrir l'incertitude inter-modèles).
MODELES_DEFAUT = ("EC-Earth3", "MPI-ESM1-2-HR", "MRI-ESM2-0", "NorESM2-MM", "GFDL-ESM4", "IPSL-CM6A-LR")
PREMIERE_ANNEE = {"historical": 1950}
DERNIERE_ANNEE = {"historical": 2014}

_NOM = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
_ANNEE = re.compile(r"_(\d{4})(?:_v[\d.]+)?\.nc$")


@dataclass(frozen=True)
class Emprise:
    ouest: float
    sud: float
    est: float
    nord: float

    def valider(self) -> None:
        if not (-180 <= self.ouest < self.est <= 180 and -90 <= self.sud < self.nord <= 90):
            raise ValueError("Emprise invalide (ouest < est, sud < nord, en degrés).")
        if (self.est - self.ouest) * (self.nord - self.sud) > 25:
            raise ValueError("Emprise trop grande pour une extraction NEX-GDDP (> 25 degrés carrés).")


@dataclass(frozen=True)
class Grille:
    """Valeurs journalières d'une variable sur une emprise, pour une année."""

    variable: str
    annee: int
    lat: np.ndarray
    lon: np.ndarray
    valeurs: np.ndarray  # (jours, lat, lon), unités converties (mm/jour, °C), NaN = manquant


def _verifier_nom(valeur: str, quoi: str) -> str:
    if not _NOM.match(valeur):
        raise ValueError(f"{quoi} invalide : {valeur!r}")
    return valeur


def dossier(modele: str, scenario: str, membre: str, variable: str) -> str:
    if scenario not in SCENARIOS:
        raise ValueError(f"Scénario inconnu : {scenario}")
    if variable not in VARIABLES:
        raise ValueError(f"Variable inconnue : {variable}")
    return "/".join([RACINE_DONNEES, _verifier_nom(modele, "Modèle"), scenario,
                     _verifier_nom(membre, "Membre"), variable])


def _titres_catalogue(client: NasaHttpClient, chemin: str) -> list[str]:
    reponse = client.request("GET", f"{THREDDS}/catalog/{chemin}/catalog.xml")
    return re.findall(r'xlink:title="([^"]+)"', reponse.text)


def membres(client: NasaHttpClient, modele: str, scenario: str) -> list[str]:
    chemin = f"{RACINE_DONNEES}/{_verifier_nom(modele, 'Modèle')}/{scenario}"
    return [t for t in _titres_catalogue(client, chemin) if _NOM.match(t)]


def lister_fichiers(client: NasaHttpClient, modele: str, scenario: str, membre: str,
                    variable: str) -> dict[int, str]:
    """``{année: urlPath}`` des fichiers annuels (la dernière version si plusieurs)."""
    reponse = client.request("GET", f"{THREDDS}/catalog/{dossier(modele, scenario, membre, variable)}/catalog.xml")
    fichiers: dict[int, str] = {}
    for nom, chemin in re.findall(r'<dataset\s+name="([^"]+\.nc)"[^>]*urlPath="([^"]+)"', reponse.text):
        trouve = _ANNEE.search(nom)
        if trouve and chemin.startswith(RACINE_DONNEES + "/"):
            annee = int(trouve.group(1))
            if annee not in fichiers or chemin > fichiers[annee]:  # « _v2.0 » > « _v1.1 » > sans version
                fichiers[annee] = chemin
    if not fichiers:
        raise NasaApiError(SERVICE, f"Aucun fichier pour {modele}/{scenario}/{membre}/{variable}.")
    return dict(sorted(fichiers.items()))


def telecharger(client: NasaHttpClient, url_path: str, variable: str, emprise: Emprise) -> bytes:
    """Découpe d'un fichier annuel sur l'emprise (NetCDF-3, toutes les dates)."""
    emprise.valider()
    if not url_path.startswith(RACINE_DONNEES + "/") or ".." in url_path:
        raise ValueError("Chemin NEX-GDDP inattendu.")
    marge = 0.13  # une demi-cellule : garde les cellules qui touchent le bord
    reponse = client.request("GET", f"{THREDDS}/ncss/grid/{url_path}", params={
        "var": variable, "north": emprise.nord + marge, "south": emprise.sud - marge,
        "west": emprise.ouest - marge, "east": emprise.est + marge,
        "horizStride": 1, "temporal": "all", "accept": "netcdf3",
    })
    if not reponse.content.startswith(b"CDF"):
        raise NasaApiError(SERVICE, "Réponse NetCDF-3 attendue.", reponse.status_code)
    return reponse.content


def lire(contenu: bytes, variable: str, annee: int) -> Grille:
    """Lit un découpage NetCDF-3 et convertit les unités (kg m-2 s-1 → mm/jour, K → °C)."""
    from scipy.io import netcdf_file

    with netcdf_file(io.BytesIO(contenu), mmap=False, maskandscale=False) as fichier:
        v = fichier.variables[variable]
        valeurs = np.array(v[:], dtype="float64")
        for attribut in ("_FillValue", "missing_value"):
            manquant = getattr(v, attribut, None)
            if manquant is not None:
                valeurs[np.isclose(valeurs, float(np.ravel(manquant)[0]))] = np.nan
        valeurs[np.abs(valeurs) > 1e19] = np.nan
        lat = np.array(fichier.variables["lat"][:], dtype="float64")
        lon = np.array(fichier.variables["lon"][:], dtype="float64")
    if valeurs.ndim != 3:
        raise NasaApiError(SERVICE, f"Dimensions inattendues pour {variable} : {valeurs.shape}")
    if variable == "pr":
        valeurs = valeurs * 86400.0
    elif variable.startswith("tas"):
        valeurs = valeurs - 273.15
    lon = np.where(lon > 180, lon - 360, lon)
    return Grille(variable=variable, annee=annee, lat=lat, lon=lon, valeurs=valeurs)


def mois_des_jours(n_jours: int) -> np.ndarray:
    """Mois (1-12) de chaque jour d'un fichier annuel, selon le calendrier du modèle
    (365 jours, 366 jours bissextile ou 360 jours de 12 x 30 jours)."""
    if n_jours == 360:
        longueurs = [30] * 12
    elif n_jours in (365, 366):
        longueurs = [31, 29 if n_jours == 366 else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    else:
        return np.minimum(np.arange(n_jours) * 12 // max(n_jours, 1) + 1, 12)
    return np.repeat(np.arange(1, 13), longueurs)


def poids_cellules(lat: np.ndarray, lon: np.ndarray, zone) -> np.ndarray:
    """Part de chaque cellule de 0,25° couverte par la zone (géométrie shapely WGS 84)."""
    from shapely.geometry import box

    pas_lat = float(np.median(np.diff(lat))) if lat.size > 1 else 0.25
    pas_lon = float(np.median(np.diff(lon))) if lon.size > 1 else 0.25
    poids = np.zeros((lat.size, lon.size))
    for i, y in enumerate(lat):
        for j, x in enumerate(lon):
            cellule = box(x - pas_lon / 2, y - pas_lat / 2, x + pas_lon / 2, y + pas_lat / 2)
            poids[i, j] = zone.intersection(cellule).area / cellule.area
    return poids


def moyenne_zone(valeurs: np.ndarray, poids: np.ndarray) -> np.ndarray:
    """Moyenne pondérée journalière sur la zone (les cellules sans donnée sont ignorées)."""
    w = np.broadcast_to(poids, valeurs.shape) * np.isfinite(valeurs)
    total = w.sum(axis=(1, 2))
    somme = np.nansum(np.nan_to_num(valeurs) * w, axis=(1, 2))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(total > 0, somme / total, math.nan)
