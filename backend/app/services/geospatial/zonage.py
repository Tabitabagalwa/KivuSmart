"""Chaîne commune « rasters → zones d'intervention » (données réelles et démonstration).

* :class:`Grille` : grille métrique UTM 35S couvrant une emprise WGS 84 ;
* :func:`analyser` : règles d'aptitude partagées (:mod:`decision_engine.aptitude`) ;
* :func:`zones_par_strategie` : polygones RNA / agroforesterie / reboisement / prioritaires,
  avec statistiques et justification chiffrée ;
* écriture GeoTIFF / GeoJSON et de la configuration lue par le frontend.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.features import rasterize, shapes
from rasterio.transform import from_origin
from scipy import ndimage
from shapely.geometry import shape
from shapely.ops import transform as shapely_transform

from ...decision_engine.aptitude import aptitude as note_aptitude
from ...decision_engine.aptitude import criteres

SCR = "EPSG:32735"
NODATA = -9999.0
VERS_UTM = Transformer.from_crs("EPSG:4326", SCR, always_xy=True)
VERS_WGS = Transformer.from_crs(SCR, "EPSG:4326", always_xy=True)

CLASSES = {
    1: ("Forêt primaire", "#14532d"),
    2: ("Forêt secondaire", "#22c55e"),
    3: ("Savane / prairie", "#d9f99d"),
    4: ("Mosaïque champs-jachère", "#f59e0b"),
    5: ("Eau", "#2563eb"),
    6: ("Bâti", "#6b7280"),
}
PALETTES = {"altitude_m": "altitude", "pente_pct": "pente", "ndvi_2015": "vegetation",
            "ndvi_2025": "vegetation", "changement_ndvi_2015_2025": "variation",
            "couverture_arboree_pct": "vegetation", "pluie_mm_an": "eau", "humidite_sol": "eau",
            "annees_brulees_2015_2025": "feux", "occupation_sol_2025": "categorique",
            "perte_foret_2001_2023": "feux"}


def utm(geometrie):
    return shapely_transform(lambda x, y, z=None: VERS_UTM.transform(x, y), geometrie)


def wgs(geometrie):
    return shapely_transform(lambda x, y, z=None: VERS_WGS.transform(x, y), geometrie)


class Grille:
    """Grille UTM 35S de pas ``res`` couvrant l'emprise (ouest, sud, est, nord) en WGS 84."""

    def __init__(self, emprise: tuple[float, float, float, float], res: float = 30.0) -> None:
        ouest, sud, est, nord = emprise
        self.res = float(res)
        xs, ys = VERS_UTM.transform([ouest, est, ouest, est], [sud, sud, nord, nord])
        self.gauche = math.floor(min(xs) / res) * res
        self.haut = math.ceil(max(ys) / res) * res
        self.largeur = int(math.ceil((max(xs) - self.gauche) / res))
        self.hauteur = int(math.ceil((self.haut - min(ys)) / res))
        self.transform = from_origin(self.gauche, self.haut, res, res)
        self.x = self.gauche + (np.arange(self.largeur) + 0.5) * res      # abscisses UTM des centres
        self.y = self.haut - (np.arange(self.hauteur) + 0.5) * res
        lon, _ = VERS_WGS.transform(self.x, np.full_like(self.x, self.y.mean()))
        self.lon = np.broadcast_to(lon, (self.hauteur, self.largeur))
        self.forme = (self.hauteur, self.largeur)

    @property
    def bornes_utm(self) -> tuple[float, float, float, float]:
        return (self.gauche, self.haut - self.hauteur * self.res, self.gauche + self.largeur * self.res, self.haut)

    def raster(self, geometries, valeur=1, all_touched: bool = False) -> np.ndarray:
        geometries = [g for g in geometries if g is not None and not g.is_empty]
        if not geometries:
            return np.zeros(self.forme, dtype=bool)
        return rasterize(((g, valeur) for g in geometries), out_shape=self.forme, transform=self.transform,
                         fill=0, dtype="uint8", all_touched=all_touched).astype(bool)

    def distance(self, masque: np.ndarray) -> np.ndarray:
        if not masque.any():
            return np.full(self.forme, 1e6, dtype="float32")
        return ndimage.distance_transform_edt(~masque, sampling=self.res).astype("float32")


# ----------------------------------------------------------------------- analyse
def analyser(r: dict[str, np.ndarray], res: float = 30.0):
    """Masques et notes d'aptitude, avec les règles partagées de decision_engine.aptitude."""
    pente = ndimage.gaussian_filter(r["pente"], 1.5 * 30.0 / res)   # pente lissée (~90 m) : zones lisibles
    v = {
        "pente": pente, "occupation": r["occupation"], "dans_pnkb": r["pnkb"], "dans_zone": r["dans_zone"],
        "d_pnkb": r["d_pnkb"], "d_champs": r["d_champs"], "d_village": r["d_village"],
        "d_route": r["d_route"], "d_riviere": r["d_riviere"], "ndvi": r["ndvi_2025"],
        "variation_ndvi": r["changement_ndvi"], "couvert": r["couvert"], "feux": r["feux"],
    }
    notes = {p: note_aptitude(p, v) for p in ("rna", "agroforesterie", "reboisement")}
    return criteres(v), notes, pente


def niveau(note: float) -> str:
    return "très élevée" if note >= 4.2 else "élevée" if note >= 3.5 else "moyenne" if note >= 2.8 else "faible"


def polygoniser(masque, g: Grille, r, note, pente, prefixe: str, intervention: str,
                justifier: Callable[[dict[str, float]], str], surface_min_ha: float = 8,
                suffixe: str = " À valider sur le terrain avec les exploitants.") -> list[dict[str, Any]]:
    masque = ndimage.binary_opening(masque, iterations=1)
    etiquettes, n = ndimage.label(masque)
    if n == 0:
        return []
    tailles = ndimage.sum(masque, etiquettes, np.arange(1, n + 1))
    gardes = [i + 1 for i, t in enumerate(tailles) if t * g.res * g.res / 10_000 >= surface_min_ha]
    if not gardes:
        return []
    etiquettes = np.where(np.isin(etiquettes, gardes), etiquettes, 0).astype(np.int32)
    stat = {nom: dict(zip(gardes, ndimage.mean(valeurs, etiquettes, gardes)))
            for nom, valeurs in {"pente": pente, "ndvi": r["ndvi_2025"], "delta": r["changement_ndvi"],
                                 "couvert": r["couvert"], "feux": r["feux"], "pnkb": r["d_pnkb"],
                                 "village": r["d_village"], "champs": r["d_champs"],
                                 "riviere": r["d_riviere"], "note": note,
                                 "altitude": r["altitude"]}.items()}
    morceaux: dict[int, list] = {}
    for geom, valeur in shapes(etiquettes, mask=etiquettes > 0, transform=g.transform, connectivity=4):
        morceaux.setdefault(int(valeur), []).append(shape(geom))
    entites = []
    for rang, (etiquette, parties) in enumerate(sorted(morceaux.items()), start=1):
        geometrie = parties[0] if len(parties) == 1 else parties[0].union(*parties[1:])
        geometrie = geometrie.simplify(20, preserve_topology=True)
        s = {k: float(v[etiquette]) for k, v in stat.items()}
        entites.append({
            "id": f"{prefixe}-{rang:03d}",
            "nom": f"{intervention} — zone {rang}",
            "intervention": intervention,
            "aptitude_sur_5": round(s["note"], 2),
            "priorite": niveau(s["note"]),
            "surface_ha": round(geometrie.area / 10_000, 1),
            "pente_moyenne_pct": round(s["pente"], 1),
            "ndvi_2025_moyen": round(s["ndvi"], 2),
            "variation_ndvi_2015_2025": round(s["delta"], 2),
            "altitude_moyenne_m": round(s["altitude"]),
            "justification": justifier(s) + suffixe,
            "geometry": wgs(geometrie),
        })
    return entites


def zones_par_strategie(masques, notes, pente, g: Grille, r, surface_min_ha: float = 8) -> dict[str, list]:
    p = lambda cle, *args: polygoniser(masques[cle], g, r, *args, surface_min_ha=surface_min_ha)  # noqa: E731
    return {
        "rna": p("rna", notes["rna"], pente, "RNA", "Régénération naturelle assistée",
                 lambda s: (f"Pente moyenne {s['pente']:.0f} % (critère 25–35 %), "
                            + ("dans le PNKB ou en lisière ; " if s["pnkb"] < 100 else f"à {s['pnkb']:.0f} m du PNKB ; ")
                            + f"NDVI 2025 {s['ndvi']:.2f}, couvert arboré {s['couvert']:.0f} %, "
                            f"{s['feux']:.1f} année(s) brûlée(s) en moyenne. Aptitude AHP {s['note']:.1f}/5.")),
        "agroforesterie": p("agroforesterie", notes["agroforesterie"], pente, "AGRO", "Agroforesterie",
                            lambda s: (f"À {s['champs']:.0f} m des champs et {s['village']:.0f} m d'un village "
                                       f"(critères ≤ 500 m et ≤ 2 km) ; pente moyenne {s['pente']:.0f} %, "
                                       f"à {s['riviere']:.0f} m d'un cours d'eau. Aptitude AHP {s['note']:.1f}/5.")),
        "reboisement": p("reboisement", notes["reboisement"], pente, "REB", "Reboisement pur",
                         lambda s: (f"Pente moyenne {s['pente']:.0f} % (critère > 35 %), hors usage agricole et "
                                    f"hors PNKB ; à {s['riviere']:.0f} m d'un cours d'eau, NDVI 2025 {s['ndvi']:.2f}. "
                                    f"Aptitude AHP {s['note']:.1f}/5.")),
        "prioritaire": p("prioritaire", notes["reboisement"], pente, "PRIO",
                         "Zone prioritaire (érosion ou dégradation récente)",
                         lambda s: (f"Variation NDVI 2015-2025 de {s['delta']:+.2f}, pente moyenne {s['pente']:.0f} % : "
                                    "champs sur versant raide ou défrichement récent, risque d'érosion élevé.")),
    }


# ---------------------------------------------------------------------- écriture
def ecrire_raster(chemin: Path, valeurs, g: Grille, dans_zone, description: str, unite: str,
                  source: str, nodata: float = NODATA, **etiquettes: str) -> None:
    chemin.parent.mkdir(parents=True, exist_ok=True)
    donnees = np.where(dans_zone & np.isfinite(valeurs), valeurs, nodata).astype("float32")
    with rasterio.open(chemin, "w", driver="GTiff", width=g.largeur, height=g.hauteur, count=1,
                       dtype="float32", crs=SCR, transform=g.transform, nodata=nodata,
                       compress="deflate", tiled=True, predictor=3) as sortie:
        sortie.write(donnees, 1)
        sortie.set_band_description(1, description)
        sortie.units = (unite,)
        sortie.update_tags(source=source, description=description, **etiquettes)


def ecrire_geojson(chemin: Path, entites, crs="EPSG:4326") -> None:
    chemin.parent.mkdir(parents=True, exist_ok=True)
    if chemin.exists():
        chemin.unlink()
    if not entites:
        print(f"  attention : aucune entité pour {chemin.name} (couche non écrite)")
        return
    gpd.GeoDataFrame(entites, geometry="geometry", crs=crs).to_file(chemin, driver="GeoJSON")


def entree_raster(racine: Path, chemin: Path, nom: str, description: str, unite: str, source: str,
                  produit: str, satellite: bool, categoriel: bool = False, afficher: bool = False,
                  periode: tuple[str, str] | None = None) -> dict[str, Any]:
    entree: dict[str, Any] = {
        "id": nom, "name": description, "path": chemin.relative_to(racine).as_posix(), "source": source,
        "satellite": satellite, "product": produit, "categorical": categoriel, "unit": unite,
        "palette": PALETTES.get(nom, "vegetation"), "show": afficher,
    }
    if periode:
        entree["period_start"], entree["period_end"] = periode
    if categoriel:
        entree["class_colors"] = {str(k): c for k, (_, c) in CLASSES.items()}
        entree["class_labels"] = {str(k): n for k, (n, _) in CLASSES.items()}
    return entree


def ecrire_json(chemin: Path, contenu: dict[str, Any]) -> None:
    chemin.parent.mkdir(parents=True, exist_ok=True)
    chemin.write_text(json.dumps(contenu, indent=2, ensure_ascii=False), encoding="utf-8")
