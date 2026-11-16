"""
Prépare les DONNÉES RÉELLES du territoire de Kalehe et la configuration du frontend.

Sources (toutes ouvertes, sans compte Google Earth Engine) :
  * Limite        : geoBoundaries COD ADM2 (Référentiel géographique commun, OCHA RDC)
  * Relief        : NASA NASADEM (30 m)                         — Planetary Computer
  * Végétation    : NASA/USGS Landsat 8-9 Collection 2 niveau 2 — NDVI médian de saison sèche
                    (juin-septembre) 2014-2016 (« 2015 ») et 2024-2025 (« 2025 »)
  * Couvert boisé : Hansen/UMD/Google/USGS/NASA Global Forest Change (Landsat, 30 m)
  * Feux          : NASA MODIS MCD64A1 v061 (surfaces brûlées mensuelles, 500 m), 2015-2025
  * Contexte      : OpenStreetMap (Overpass) — PNKB, localités, routes, cours d'eau, champs

L'occupation du sol est déduite par règles explicites (couvert Hansen, NDVI Landsat, eau
Hansen, champs et localités OSM) : c'est une classification indicative à valider sur le terrain.

Sorties :
  data/reel/rasters/*.tif, data/reel/vecteurs/*.geojson, data/reel/rapport_qualite.json
  config/study_area.json, config/map_layers.json (le frontend les utilise automatiquement)

Utilisation (depuis la racine du projet) :
    python scripts/preparer_donnees_reelles.py                  # 30 m (long : 20-40 min)
    python scripts/preparer_donnees_reelles.py --resolution 90  # essai rapide
Les téléchargements sont mis en cache dans data/raw/ : une relance reprend là où elle s'est arrêtée.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from rasterio.enums import Resampling
from rasterio.features import shapes
from scipy import ndimage
from shapely.geometry import LineString, MultiPolygon, Point, Polygon, shape
from shapely.ops import linemerge, polygonize, unary_union

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE))
from KivuSmart.backend.app.services.data_ingestion import sources_ouvertes as so  # noqa: E402
from KivuSmart.backend.app.services.geospatial import zonage as zn  # noqa: E402
from KivuSmart.backend.app.services.planetary_computer import stac  # noqa: E402

REEL = RACINE / "data" / "reel"
CACHE = RACINE / "data" / "raw" / "reel_cache"
CONFIG = RACINE / "config"
LIMITE = REEL / "vecteurs" / "limite_kalehe.geojson"

HANSEN = ("GFC-2024-v1.12", "GFC-2023-v1.11")
HANSEN_TUILE = "00N_020E"
EPOQUES = {"2015": ("2014-06-01/2016-09-30", (6, 7, 8, 9)), "2025": ("2024-06-01/2025-09-30", (6, 7, 8, 9))}
SCENES_PAR_TRACE = 4
SOURCES = {
    "limite": "geoBoundaries gbOpen COD ADM2 (OCHA RDC, Référentiel géographique commun)",
    "nasadem": "NASA NASADEM (LP DAAC), via Microsoft Planetary Computer",
    "landsat": "NASA/USGS Landsat 8-9 Collection 2 niveau 2, via Microsoft Planetary Computer",
    "hansen": "Hansen/UMD/Google/USGS/NASA Global Forest Change (Landsat)",
    "modis": "NASA MODIS MCD64A1 v061 (LP DAAC), via Microsoft Planetary Computer",
    "osm": "© contributeurs OpenStreetMap (ODbL), via Overpass",
    "derive": "Dérivé NASA (Landsat, NASADEM) + Hansen GFC + OpenStreetMap — règles Kivu Smart",
}


def etape(titre: str) -> float:
    print(f"\n== {titre}", flush=True)
    return time.time()


def fin(t0: float) -> None:
    print(f"   ({time.time() - t0:.0f} s)", flush=True)


def en_cache(nom: str, calcul, forcer: bool = False):
    """Met en cache le résultat (dict de tableaux numpy) d'une étape coûteuse."""
    chemin = CACHE / f"{nom}.npz"
    if chemin.is_file() and not forcer:
        with np.load(chemin) as f:
            return {k: f[k] for k in f.files}
    resultat = calcul()
    chemin.parent.mkdir(parents=True, exist_ok=True)
    temporaire = chemin.with_name(chemin.stem + ".part.npz")
    np.savez_compressed(temporaire, **resultat)
    temporaire.replace(chemin)
    return resultat


# ------------------------------------------------------------------------ limite
def limite():
    if not LIMITE.is_file():
        with so.client() as http:
            entite = so.limite_geoboundaries(http)
        LIMITE.parent.mkdir(parents=True, exist_ok=True)
        LIMITE.write_text(json.dumps({"type": "FeatureCollection", "features": [entite]}, ensure_ascii=False),
                          encoding="utf-8")
    collection = json.loads(LIMITE.read_text(encoding="utf-8"))
    entite = collection["features"][0]
    return shape(entite["geometry"]).buffer(0), entite["properties"]


# ---------------------------------------------------------------- OpenStreetMap
def osm(emprise, forcer: bool = False) -> dict[str, dict]:
    bb = so.emprise_overpass(*emprise)
    requetes = {
        "localites": f'[out:json][timeout:300];node["place"~"^(city|town|village|hamlet|suburb|neighbourhood)$"]({bb});out;',
        "routes": (f'[out:json][timeout:300];way["highway"~"^(trunk|primary|secondary|tertiary|unclassified|'
                   f'residential|track)$"]({bb});out geom;'),
        "cours_eau": f'[out:json][timeout:300];way["waterway"~"^(river|stream)$"]({bb});out geom;',
        "champs": (f'[out:json][timeout:300];(way["landuse"~"^(farmland|farm|orchard|meadow|farmyard)$"]({bb});'
                   f'relation["landuse"~"^(farmland|farm|orchard)$"]({bb}););out geom;'),
        "pnkb": (f'[out:json][timeout:300];relation["name"~"Kahuzi",i]["boundary"~"^(protected_area|national_park)$"]'
                 f'({bb});out geom;'),
    }
    resultats = {}
    with so.client() as http:
        for nom, requete in requetes.items():
            chemin = CACHE / "osm" / f"{nom}.json"
            if chemin.is_file() and not forcer:
                resultats[nom] = json.loads(chemin.read_text(encoding="utf-8"))
                continue
            t0 = time.time()
            donnees = so.overpass(http, requete)
            chemin.parent.mkdir(parents=True, exist_ok=True)
            chemin.write_text(json.dumps(donnees), encoding="utf-8")
            resultats[nom] = donnees
            print(f"   OSM {nom:<10}: {len(donnees.get('elements', [])):>6} objets ({time.time() - t0:.0f} s)", flush=True)
    return resultats


def _ligne(element) -> LineString | None:
    points = [(p["lon"], p["lat"]) for p in element.get("geometry") or [] if p]
    return LineString(points) if len(points) >= 2 else None


def _polygone_relation(element):
    exterieurs, interieurs = [], []
    for membre in element.get("members", []):
        ligne = _ligne(membre)
        if ligne is not None:
            (interieurs if membre.get("role") == "inner" else exterieurs).append(ligne)
    polygones = list(polygonize(linemerge(exterieurs))) if exterieurs else []
    if not polygones:
        return None
    forme = unary_union(polygones)
    if interieurs:
        trous = list(polygonize(linemerge(interieurs)))
        if trous:
            forme = forme.difference(unary_union(trous))
    return forme.buffer(0)


def vecteurs_osm(brut: dict[str, dict]):
    localites = [{"nom": e.get("tags", {}).get("name", "Localité"), "type": e.get("tags", {}).get("place"),
                  "geometry": Point(e["lon"], e["lat"])} for e in brut["localites"]["elements"] if e["type"] == "node"]
    routes = [{"nom": e.get("tags", {}).get("name") or e.get("tags", {}).get("ref") or "Route",
               "type": e.get("tags", {}).get("highway"), "geometry": g}
              for e in brut["routes"]["elements"] if (g := _ligne(e)) is not None]
    cours_eau = [{"nom": e.get("tags", {}).get("name") or ("Rivière" if e.get("tags", {}).get("waterway") == "river"
                                                           else "Ruisseau"),
                  "type": e.get("tags", {}).get("waterway"), "geometry": g}
                 for e in brut["cours_eau"]["elements"] if (g := _ligne(e)) is not None]
    champs = []
    for e in brut["champs"]["elements"]:
        if e["type"] == "way":
            ligne = _ligne(e)
            if ligne is not None and len(ligne.coords) >= 4 and ligne.is_ring:
                champs.append(Polygon(ligne.coords).buffer(0))
        elif e["type"] == "relation" and (forme := _polygone_relation(e)) is not None:
            champs.append(forme)
    parcs = [f for e in brut["pnkb"]["elements"] if e["type"] == "relation" and (f := _polygone_relation(e)) is not None]
    pnkb = unary_union(parcs) if parcs else None
    return localites, routes, cours_eau, champs, pnkb


# --------------------------------------------------------------------- rasters
def nasadem(cat: stac.Catalogue, g: zn.Grille, bbox):
    items = cat.rechercher("nasadem", bbox, "2000-01-01/2001-01-01")
    if not items:
        raise SystemExit("NASADEM : aucune tuile trouvée.")
    altitude = np.full(g.forme, np.nan, dtype="float32")
    for item in items:
        valeurs = stac.lire_sur_grille(cat.href_signe(item, "elevation"), zn.SCR, g.transform, g.largeur, g.hauteur,
                                       Resampling.bilinear, nodata_source=-32768)
        altitude = np.where(np.isnan(altitude), valeurs, altitude)
    print(f"   NASADEM : {len(items)} tuile(s), altitude {np.nanmin(altitude):.0f}-{np.nanmax(altitude):.0f} m")
    return {"altitude": altitude, "tuiles": np.array(len(items))}


def hansen(g: zn.Grille):
    def couche(version, nom):
        url = (f"https://storage.googleapis.com/earthenginepartners-hansen/{version}/"
               f"Hansen_{version}_{nom}_{HANSEN_TUILE}.tif")
        return stac.lire_sur_grille(url, zn.SCR, g.transform, g.largeur, g.hauteur, Resampling.nearest,
                                    dtype="uint8", nodata_source=255)
    derniere = None
    for version in HANSEN:
        try:
            couvert = couche(version, "treecover2000")
            perte = couche(version, "lossyear")
            masque = couche(version, "datamask")
            print(f"   Hansen {version} : couvert 2000 moyen {couvert.mean():.0f} %")
            return {"couvert_2000": couvert, "perte": perte, "masque": masque, "version": np.array(version)}
        except Exception as exc:  # version absente : on essaie la précédente
            derniere = exc
            print(f"   Hansen {version} indisponible ({type(exc).__name__})")
    raise SystemExit(f"Hansen GFC indisponible : {derniere}")


def ndvi_landsat(cat: stac.Catalogue, g: zn.Grille, bbox, epoque: str):
    intervalle, mois = EPOQUES[epoque]
    items = cat.rechercher("landsat-c2-l2", bbox, intervalle, query={
        "eo:cloud_cover": {"lt": 70}, "platform": {"in": ["landsat-8", "landsat-9"]}})
    par_trace = defaultdict(list)
    for item in items:
        if int(item["properties"]["datetime"][5:7]) in mois:
            p = item["properties"]
            par_trace[(p.get("landsat:wrs_path"), p.get("landsat:wrs_row"))].append(item)
    choisis = [i for liste in par_trace.values()
               for i in sorted(liste, key=lambda x: x["properties"]["eo:cloud_cover"])[:SCENES_PAR_TRACE]]
    print(f"   Landsat {epoque} : {len(items)} scène(s) trouvée(s), {len(choisis)} retenue(s) "
          f"sur {len(par_trace)} trace(s)", flush=True)
    pile = []
    for item in choisis:
        def scene(item=item):
            qa = stac.lire_sur_grille(cat.href_signe(item, "qa_pixel"), zn.SCR, g.transform, g.largeur, g.hauteur,
                                      Resampling.nearest, dtype="uint16", nodata_source=1)
            rouge = stac.lire_sur_grille(cat.href_signe(item, "red"), zn.SCR, g.transform, g.largeur, g.hauteur,
                                         Resampling.bilinear, nodata_source=0)
            pir = stac.lire_sur_grille(cat.href_signe(item, "nir08"), zn.SCR, g.transform, g.largeur, g.hauteur,
                                       Resampling.bilinear, nodata_source=0)
            clair = (qa > 1) & ((qa & 0b11111) == 0)          # ni remplissage, nuage dilaté, cirrus, nuage, ombre
            rouge, pir = rouge * 2.75e-5 - 0.2, pir * 2.75e-5 - 0.2
            with np.errstate(invalid="ignore", divide="ignore"):
                ndvi = (pir - rouge) / (pir + rouge)
            ndvi = np.where(clair & np.isfinite(ndvi) & (rouge > -0.05) & (pir > 0), ndvi, np.nan)
            return {"ndvi": np.clip(ndvi, -1, 1).astype("float16")}
        t0 = time.time()
        try:
            n = en_cache(f"landsat_{item['id']}_{g.res:.0f}m", scene)["ndvi"]
        except Exception as exc:
            print(f"     {item['id']} ignorée ({type(exc).__name__}: {str(exc)[:100]})")
            continue
        pile.append(n)
        print(f"     {item['id']} nuages {item['properties']['eo:cloud_cover']:.0f} %, "
              f"{np.isfinite(n).mean() * 100:.0f} % de pixels clairs ({time.time() - t0:.0f} s)", flush=True)
    if not pile:
        raise SystemExit(f"Landsat {epoque} : aucune scène lisible.")
    mediane = np.full(g.forme, np.nan, dtype="float32")
    pas = 256
    for debut in range(0, g.hauteur, pas):
        bloc = np.stack([p[debut:debut + pas].astype("float32") for p in pile])
        with np.errstate(all="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                mediane[debut:debut + pas] = np.nanmedian(bloc, axis=0)
        nombre = np.isfinite(bloc).sum(axis=0)
        mediane[debut:debut + pas][nombre == 0] = np.nan
    return {"ndvi": mediane, "scenes": np.array(len(pile)),
            "identifiants": np.array([i["id"] for i in choisis])}


def feux_modis(cat: stac.Catalogue, g: zn.Grille, bbox):
    items = cat.rechercher("modis-64A1-061", bbox, "2015-01-01/2025-12-31", limite=1000)
    facteur = max(1, int(round(450 / g.res)))   # lecture sur une grille ~450 m (produit à 500 m)
    import rasterio.transform as rt
    largeur, hauteur = -(-g.largeur // facteur), -(-g.hauteur // facteur)
    transform = rt.from_origin(g.gauche, g.haut, g.res * facteur, g.res * facteur)
    brule_par_annee: dict[int, np.ndarray] = {}
    for n, item in enumerate(sorted(items, key=lambda i: i["properties"]["datetime"]), start=1):
        annee = int(item["properties"]["datetime"][:4])
        try:
            jour = stac.lire_sur_grille(cat.href_signe(item, "Burn_Date"), zn.SCR, transform, largeur, hauteur,
                                        Resampling.nearest, dtype="int16")
        except Exception as exc:
            print(f"     {item['id']} ignorée ({type(exc).__name__})")
            continue
        brule_par_annee.setdefault(annee, np.zeros((hauteur, largeur), dtype=bool))
        brule_par_annee[annee] |= jour > 0
        if n % 40 == 0:
            print(f"     MODIS : {n}/{len(items)} mois lus", flush=True)
    annees = sum(m.astype("uint8") for m in brule_par_annee.values()) if brule_par_annee else \
        np.zeros((hauteur, largeur), dtype="uint8")
    plein = np.repeat(np.repeat(annees, facteur, axis=0), facteur, axis=1)[: g.hauteur, : g.largeur]
    print(f"   MODIS MCD64A1 : {len(items)} granule(s) mensuel(s), {len(brule_par_annee)} année(s)")
    return {"feux": plein.astype("float32"), "annees": np.array(sorted(brule_par_annee))}


# -------------------------------------------------------------- occupation du sol
def classer_occupation(couvert, ndvi, eau, pnkb, d_village, d_urbain, champs_osm):
    """Classes 1-6 du document (règles explicites, à valider sur le terrain)."""
    occ = np.full(couvert.shape, 3, dtype=np.int16)                       # savane / prairie par défaut
    ndvi0 = np.nan_to_num(ndvi, nan=0.5)
    occ[(couvert >= 30) | ((couvert >= 15) & (ndvi0 >= 0.65))] = 2         # forêt secondaire / dégradée
    occ[(couvert >= 70) & (ndvi0 >= 0.70)] = 1                             # forêt dense
    champs = champs_osm | ((couvert < 30) & (ndvi0 < 0.62) & (d_village <= 1500) & ~pnkb)
    occ[champs & (occ != 1)] = 4                                           # mosaïque champs-jachère
    occ[((d_urbain <= 600) & (ndvi0 < 0.40)) | ((d_village <= 90) & (ndvi0 < 0.35))] = 6   # bâti
    occ[eau] = 5                                                           # eau
    return occ


def polygones_masque(masque, g: zn.Grille, simplification_m: float):
    formes = [shape(geom) for geom, v in shapes(masque.astype("uint8"), mask=masque, transform=g.transform) if v]
    return [zn.wgs(f.simplify(simplification_m)) for f in formes if f.area >= 250_000]


# ------------------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser(description="Données réelles du territoire de Kalehe.")
    parser.add_argument("--resolution", type=float, default=30.0, help="pas de la grille en mètres (30 par défaut)")
    parser.add_argument("--forcer", action="store_true", help="ignore le cache et retélécharge")
    parser.add_argument("--surface-min", type=float, default=10.0, help="surface minimale d'une zone (ha)")
    args = parser.parse_args()
    debut_total = time.time()

    t0 = etape("1. Limite du territoire de Kalehe (geoBoundaries)")
    zone, proprietes_limite = limite()
    emprise = zone.bounds
    g = zn.Grille(emprise, args.resolution)
    dans_zone = g.raster([zn.utm(zone)])
    print(f"   {zn.utm(zone).area / 1e6:,.0f} km², grille {g.largeur} x {g.hauteur} à {g.res:.0f} m")
    fin(t0)

    t0 = etape("2. Contexte OpenStreetMap (Overpass)")
    brut = osm(emprise, args.forcer)
    localites, routes, cours_eau, champs, pnkb = vecteurs_osm(brut)
    print(f"   {len(localites)} localités, {len(routes)} tronçons de route, {len(cours_eau)} cours d'eau, "
          f"{len(champs)} champs cartographiés, PNKB {'trouvé' if pnkb is not None else 'ABSENT'}")
    if pnkb is None:
        raise SystemExit("Limite du PNKB introuvable dans OpenStreetMap.")
    fin(t0)

    bbox = (emprise[0] - 0.01, emprise[1] - 0.01, emprise[2] + 0.01, emprise[3] + 0.01)
    suffixe = f"{g.res:.0f}m"
    with stac.Catalogue() as cat:
        t0 = etape("3. Relief NASA NASADEM")
        dem = en_cache(f"nasadem_{suffixe}", lambda: nasadem(cat, g, bbox), args.forcer)
        fin(t0)
        t0 = etape("4. Couvert forestier Hansen GFC (Landsat)")
        gfc = en_cache(f"hansen_{suffixe}", lambda: hansen(g), args.forcer)
        fin(t0)
        ndvi, scenes = {}, {}
        for epoque in EPOQUES:
            t0 = etape(f"5. NDVI Landsat {epoque} (saison sèche, médiane)")
            resultat = en_cache(f"ndvi_{epoque}_{suffixe}", lambda e=epoque: ndvi_landsat(cat, g, bbox, e), args.forcer)
            ndvi[epoque], scenes[epoque] = resultat["ndvi"], resultat["identifiants"].tolist()
            fin(t0)
        t0 = etape("6. Feux NASA MODIS MCD64A1 (2015-2025)")
        feux = en_cache(f"modis_feux_{suffixe}", lambda: feux_modis(cat, g, bbox), args.forcer)
        fin(t0)

    t0 = etape("7. Variables dérivées et occupation du sol")
    altitude = dem["altitude"].astype("float32")
    altitude = np.where(np.isnan(altitude), np.nanmean(altitude), altitude)
    dzdy, dzdx = np.gradient(altitude.astype("float64"), g.res, g.res)
    pente = (np.hypot(dzdx, dzdy) * 100.0).astype("float32")
    orientation = (np.degrees(np.arctan2(-dzdx, dzdy)) % 360.0).astype("float32")
    del dzdx, dzdy

    m_pnkb = g.raster([zn.utm(pnkb)])
    eau = (gfc["masque"] == 2) | g.raster([zn.utm(r["geometry"]) for r in cours_eau if r["type"] == "river"])
    perte = gfc["perte"]
    couvert = np.where(perte > 0, 0, gfc["couvert_2000"]).astype("float32")   # couvert 2000 moins les pertes
    urbains = [zn.utm(l["geometry"]) for l in localites if l["type"] in ("city", "town", "suburb")]
    villages = [zn.utm(l["geometry"]).buffer(g.res) for l in localites]
    d_village = g.distance(g.raster(villages, all_touched=True))
    d_urbain = g.distance(g.raster([p.buffer(g.res) for p in urbains], all_touched=True))
    champs_osm = g.raster([zn.utm(c) for c in champs])
    occupation = classer_occupation(couvert, ndvi["2025"], eau, m_pnkb, d_village, d_urbain, champs_osm)
    n2015 = np.where(np.isnan(ndvi["2015"]), ndvi["2025"], ndvi["2015"])
    n2025 = np.where(np.isnan(ndvi["2025"]), n2015, ndvi["2025"])
    variation = np.where(np.isfinite(ndvi["2015"]) & np.isfinite(ndvi["2025"]), n2025 - n2015, 0.0).astype("float32")

    r = {
        "dans_zone": dans_zone, "pnkb": m_pnkb, "occupation": occupation, "altitude": altitude,
        "pente": pente, "ndvi_2025": np.nan_to_num(n2025, nan=0.5).astype("float32"),
        "changement_ndvi": variation, "couvert": couvert, "feux": feux["feux"],
        "d_pnkb": g.distance(m_pnkb), "d_village": d_village,
        "d_route": g.distance(g.raster([zn.utm(x["geometry"]) for x in routes], all_touched=True)),
        "d_riviere": g.distance(g.raster([zn.utm(x["geometry"]) for x in cours_eau], all_touched=True)),
        "d_champs": g.distance(occupation == 4),
    }
    fin(t0)

    t0 = etape("8. Règles d'aptitude et zones d'intervention")
    masques, notes, pente_lissee = zn.analyser(r, g.res)
    zones = zn.zones_par_strategie(masques, notes, pente_lissee, g, r, surface_min_ha=args.surface_min)
    v = REEL / "vecteurs"
    for nom, entites in zones.items():
        zn.ecrire_geojson(v / f"zones_{nom}.geojson", entites)
        print(f"   zones {nom:<15}: {len(entites):>4} polygone(s), "
              f"{sum(e['surface_ha'] for e in entites):>9,.0f} ha".replace(",", " "))
    fin(t0)

    t0 = etape("9. Écriture des couches et de la configuration")
    zone_wgs = zone
    vue = zone_wgs.buffer(0.05)
    zn.ecrire_geojson(v / "pnkb.geojson", [{"id": "pnkb", "nom": "Parc national de Kahuzi-Biega",
                                            "type": "Aire protégée", "geometry": pnkb.intersection(vue)}])
    lacs = polygones_masque(gfc["masque"] == 2, g, 60)
    zn.ecrire_geojson(v / "eau.geojson", [{"id": f"eau-{i}", "nom": "Lac Kivu" if p.area > 0.05 else "Plan d'eau",
                                           "type": "Eau", "geometry": p} for i, p in enumerate(lacs, 1)])
    zn.ecrire_geojson(v / "localites.geojson", [
        {"id": f"loc-{i:04d}", "nom": l["nom"], "type": {"city": "Ville", "town": "Centre", "suburb": "Quartier"}
         .get(l["type"], "Village"), "geometry": l["geometry"]}
        for i, l in enumerate(localites, 1) if zone_wgs.buffer(0.01).contains(l["geometry"])
        and l["type"] in ("city", "town", "village", "suburb")])
    principales = ("trunk", "primary", "secondary", "tertiary", "unclassified")
    zn.ecrire_geojson(v / "routes.geojson", [
        {"id": f"route-{i:05d}", "nom": x["nom"], "type": x["type"], "geometry": x["geometry"].simplify(0.0002)}
        for i, x in enumerate(routes, 1) if x["type"] in principales and x["geometry"].intersects(vue)])
    zn.ecrire_geojson(v / "cours_eau.geojson", [
        {"id": f"eau-{i:05d}", "nom": x["nom"], "type": "Rivière" if x["type"] == "river" else "Ruisseau",
         "geometry": x["geometry"].simplify(0.0002)}
        for i, x in enumerate(cours_eau, 1) if x["geometry"].intersects(vue)])

    rr = REEL / "rasters"
    periode = ("2015-06-01", "2025-09-30")
    rasters = [
        ("altitude_m", altitude, "Altitude (NASADEM)", "m", SOURCES["nasadem"], "NASADEM_HGT v001", True, False),
        ("pente_pct", pente, "Pente (NASADEM)", "%", SOURCES["nasadem"], "NASADEM_HGT v001", True, False),
        ("ndvi_2015", n2015, "NDVI 2015 (Landsat, saison sèche)", "sans unité", SOURCES["landsat"],
         "Landsat 8 OLI C2 L2 (SR_B4, SR_B5)", True, False),
        ("ndvi_2025", n2025, "NDVI 2025 (Landsat, saison sèche)", "sans unité", SOURCES["landsat"],
         "Landsat 8-9 OLI C2 L2 (SR_B4, SR_B5)", True, False),
        ("changement_ndvi_2015_2025", variation, "Variation NDVI 2015-2025 (Landsat)", "sans unité",
         SOURCES["landsat"], "Landsat 8-9 OLI C2 L2", True, False),
        ("couverture_arboree_pct", couvert, "Couvert arboré (Hansen GFC, pertes déduites)", "%", SOURCES["hansen"],
         f"Global Forest Change {gfc['version']}", True, False),
        ("perte_foret_2001_2023", np.where(perte > 0, 2000.0 + perte, np.nan), "Année de perte forestière (Hansen GFC)",
         "année", SOURCES["hansen"], f"Global Forest Change {gfc['version']} lossyear", True, False),
        ("annees_brulees_2015_2025", feux["feux"], "Fréquence des feux 2015-2025 (MODIS)", "années",
         SOURCES["modis"], "MCD64A1 v061 Burn_Date", True, False),
        ("occupation_sol_2025", occupation, "Occupation du sol 2025 (classification par règles)", "classe",
         SOURCES["derive"], "Règles Kivu Smart (Hansen GFC + Landsat + OSM)", False, True),
    ]
    entrees = []
    for nom, valeurs, description, unite, source, produit, satellite, categoriel in rasters:
        chemin = rr / f"{nom}.tif"
        zn.ecrire_raster(chemin, valeurs, g, dans_zone, description, unite, source, produit=produit)
        entrees.append(zn.entree_raster(RACINE, chemin, nom, description, unite, source, produit, satellite,
                                        categoriel, afficher=nom == "occupation_sol_2025", periode=periode))

    chemin = lambda p: p.relative_to(RACINE).as_posix()  # noqa: E731
    zn.ecrire_json(CONFIG / "study_area.json", {
        "name": "Territoire de Kalehe",
        "country": "République démocratique du Congo",
        "boundary_path": chemin(LIMITE),
        "boundary_source": SOURCES["limite"],
        "boundary_source_crs": "EPSG:4326",
        "target_crs": zn.SCR,
        "target_resolution_m": g.res,
    })
    couche = lambda nom, fichier, role, source, satellite=False: {  # noqa: E731
        "name": nom, "path": chemin(v / fichier), "role": role, "source": source, "satellite": satellite,
        "period_start": periode[0], "period_end": periode[1]}
    couches = [
        couche("Parc national de Kahuzi-Biega", "pnkb.geojson", "environmental", SOURCES["osm"]),
        couche("Lac Kivu et plans d'eau", "eau.geojson", "environmental", SOURCES["hansen"]),
        couche("Cours d'eau", "cours_eau.geojson", "environmental", SOURCES["osm"]),
        couche("Routes", "routes.geojson", "environmental", SOURCES["osm"]),
        couche("Localités", "localites.geojson", "environmental", SOURCES["osm"]),
        couche("Zones prioritaires", "zones_prioritaire.geojson", "priority", SOURCES["derive"]),
        couche("Reboisement pur", "zones_reboisement.geojson", "forestry", SOURCES["derive"]),
        couche("Régénération naturelle assistée", "zones_rna.geojson", "rna", SOURCES["derive"]),
        couche("Agroforesterie", "zones_agroforesterie.geojson", "agroforestry", SOURCES["derive"]),
    ]
    couches = [c for c in couches if (RACINE / c["path"]).is_file()]
    zn.ecrire_json(CONFIG / "map_layers.json", {"layers": couches, "rasters": entrees})

    surfaces = {zn.CLASSES[k][0]: round(float(((occupation == k) & dans_zone).sum()) * g.res ** 2 / 1e4)
                for k in zn.CLASSES}
    zn.ecrire_json(REEL / "rapport_qualite.json", {
        "genere_le": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "zone": "Territoire de Kalehe", "surface_km2": round(zn.utm(zone).area / 1e6, 1),
        "resolution_m": g.res, "grille": [g.largeur, g.hauteur],
        "sources": SOURCES, "limite": proprietes_limite,
        "nasadem_tuiles": int(dem["tuiles"]), "hansen_version": str(gfc["version"]),
        "landsat_scenes": scenes,
        "ndvi_pixels_valides_pct": {e: round(float(np.isfinite(ndvi[e])[dans_zone].mean() * 100), 1) for e in ndvi},
        "modis_annees": [int(a) for a in feux["annees"]],
        "osm": {"localites": len(localites), "routes": len(routes), "cours_eau": len(cours_eau),
                "champs_cartographies": len(champs)},
        "occupation_sol_ha": surfaces,
        "zones": {k: {"nombre": len(e), "surface_ha": round(sum(x["surface_ha"] for x in e))} for k, e in zones.items()},
        "limites": [
            "Occupation du sol obtenue par règles (couvert Hansen, NDVI Landsat, OSM) : non validée sur le terrain.",
            "Couvert arboré = couvert 2000 moins les pertes Hansen ; les repousses après 2000 ne sont pas comptées.",
            "OpenStreetMap est incomplet en milieu rural : distances aux villages et aux champs sous-estimées "
            "ou surestimées localement.",
            "Feux MODIS à 500 m : les petits brûlis agricoles ne sont pas détectés.",
        ],
    })
    fin(t0)
    print(f"\nTerminé en {(time.time() - debut_total) / 60:.1f} min. Données : {REEL}")
    print("Le frontend utilisera automatiquement ces données (config/map_layers.json).")


if __name__ == "__main__":
    main()
