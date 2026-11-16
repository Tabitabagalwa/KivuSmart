"""
Génère un jeu de données SIMULÉES pour tester l'affichage de la plateforme Kivu Smart.

Rien ici ne provient de la NASA : le relief, la végétation, les villages, le parc et
les zones de restauration sont synthétiques (graine aléatoire fixe, donc reproductibles).
Les règles appliquées sont celles du document « Facteurs de reboisement et API NASA » :

  * RNA            : pente 25–35 % ET proximité de l'aire protégée (≤ 3 km), hors forêt primaire
  * Agroforesterie : à ≤ 500 m des champs ET ≤ 2 km d'un village, pente < 25 %, hors PNKB
  * Reboisement pur: pente > 35 %, hors usage agricole, hors PNKB
  * Exclusions     : eau et bâti ; facteurs complémentaires notés 1–5 et pondérés (AHP)

Sorties :
  data/simulated/rasters/*.tif     GeoTIFF mono-bande, UTM 35S, 30 m, nodata -9999
  data/simulated/vecteurs/*.geojson
  config/simulation/study_area.json et map_layers.json (configuration du frontend)

Utilisation (depuis la racine du projet) :
    python scripts/generer_donnees_simulees.py
    $env:KIVU_SIMULATION = "1"; streamlit run app.py

Pour revenir aux vraies données : supprimez la variable KIVU_SIMULATION.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from rasterio.features import geometry_mask
from scipy import ndimage
from shapely.geometry import LineString, Point, Polygon

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE))
from KivuSmart.backend.app.services.geospatial import zonage as zn  # noqa: E402
from KivuSmart.backend.app.services.geospatial.zonage import CLASSES, SCR, analyser, utm, wgs  # noqa: E402,F401

SORTIE = RACINE / "data" / "simulated"
CONFIG = RACINE / "config" / "simulation"

RES = 30.0
EMPRISE = (28.70, -2.42, 28.92, -2.22)          # ouest, sud, est, nord (WGS 84)
SOURCE = "Jeu de démonstration (données synthétiques)"


def Grille() -> zn.Grille:  # noqa: N802 (compatibilité avec l'ancienne classe locale)
    return zn.Grille(EMPRISE, RES)


def bruit(rng, forme, echelle_m, ecart=1.0) -> np.ndarray:
    """Champ aléatoire lissé (structure spatiale à l'échelle donnée), écart-type ``ecart``."""
    champ = ndimage.gaussian_filter(rng.normal(size=forme), echelle_m / RES)
    return champ / (champ.std() or 1.0) * ecart


def ligne_sinueuse(rng, depart, arrivee, n=40, amplitude=0.006):
    t = np.linspace(0, 1, n)
    lon = depart[0] + (arrivee[0] - depart[0]) * t
    lat = depart[1] + (arrivee[1] - depart[1]) * t
    phase = rng.uniform(0, 2 * math.pi)
    lat = lat + amplitude * np.sin(2 * math.pi * 2.5 * t + phase) * np.sin(math.pi * t)
    return LineString(zip(lon, lat))


# ------------------------------------------------------------------------ vecteurs
def construire_vecteurs(rng):
    ouest, sud, est, nord = EMPRISE
    angles = np.linspace(0, 2 * math.pi, 15, endpoint=False)
    centre = ((ouest + est) / 2 - 0.01, (sud + nord) / 2)
    rayon_lon, rayon_lat = 0.098, 0.088
    zone = Polygon([
        (centre[0] + rayon_lon * (1 + rng.uniform(-0.08, 0.06)) * math.cos(a),
         centre[1] + rayon_lat * (1 + rng.uniform(-0.08, 0.06)) * math.sin(a))
        for a in angles
    ]).buffer(0)

    lats = np.linspace(nord + 0.01, sud - 0.01, 25)
    rive = [(28.888 + 0.008 * math.sin(i / 3.0), lat) for i, lat in enumerate(lats)]
    lac = Polygon(rive + [(est + 0.02, sud - 0.01), (est + 0.02, nord + 0.01)])
    lisiere = [(28.733 + 0.010 * math.sin(i / 2.2) + rng.uniform(-0.002, 0.002), lat)
               for i, lat in enumerate(lats)]
    pnkb = Polygon([(ouest - 0.02, nord + 0.01)] + lisiere + [(ouest - 0.02, sud - 0.01)]).buffer(0)

    villages = []
    while len(villages) < 16:
        point = Point(rng.uniform(28.752, 28.872), rng.uniform(sud + 0.025, nord - 0.025))
        if zone.contains(point) and not lac.buffer(0.004).contains(point) and \
                all(point.distance(v) > 0.018 for v in villages):
            villages.append(point)

    # Routes : arbre couvrant minimal entre villages + route côtière nord-sud.
    reliees, routes = [0], []
    while len(reliees) < len(villages):
        i, j = min(((i, j) for i in reliees for j in range(len(villages)) if j not in reliees),
                   key=lambda p: villages[p[0]].distance(villages[p[1]]))
        routes.append(ligne_sinueuse(rng, villages[i].coords[0], villages[j].coords[0], 12, 0.0015))
        reliees.append(j)
    routes.append(LineString([(28.878 + 0.005 * math.sin(k / 2), lat) for k, lat in enumerate(lats)]))

    rivieres = [ligne_sinueuse(rng, (ouest - 0.01, lat0), (28.89, lat0 + rng.uniform(-0.015, 0.015)))
                for lat0 in (-2.255, -2.300, -2.345, -2.395)]
    return zone, lac, pnkb, villages, routes, rivieres


# -------------------------------------------------------------------------- rasters
def construire_rasters(rng, g: Grille, zone, lac, pnkb, villages, routes, rivieres):
    dans_zone = geometry_mask([utm(zone).__geo_interface__], g.forme, g.transform, invert=True)
    m_lac = g.raster([utm(lac)])
    m_pnkb = g.raster([utm(pnkb)])
    d_riviere = g.distance(g.raster([utm(r) for r in rivieres]))
    d_route = g.distance(g.raster([utm(r) for r in routes]))
    d_village = g.distance(g.raster([utm(v).buffer(60) for v in villages]))
    d_pnkb = g.distance(m_pnkb)

    # Relief : plateau qui monte vers l'ouest (massif du Kahuzi), collines, vallées.
    ouest_norm = np.clip((28.89 - g.lon) / 0.19, 0, 1)
    altitude = (1465 + 1250 * ouest_norm ** 1.3
                + bruit(rng, g.forme, 2500, 250) + bruit(rng, g.forme, 800, 220)
                + bruit(rng, g.forme, 300, 40)
                - 110 * np.exp(-d_riviere / 350))
    altitude = np.where(m_lac, 1462.0, np.maximum(altitude, 1465.0))
    dzdy, dzdx = np.gradient(altitude, RES, RES)
    pente = np.hypot(dzdx, dzdy) * 100.0
    orientation = np.degrees(np.arctan2(-dzdx, dzdy)) % 360.0

    # Occupation du sol (ordre de priorité : eau, bâti, forêt du parc, champs, reste).
    alea = bruit(rng, g.forme, 400)
    occupation = np.full(g.forme, 3, dtype=np.int16)
    occupation[(d_pnkb < 1800) & (alea > -0.3)] = 2
    occupation[(altitude > 2350) & (alea > 0.6)] = 2
    champs = ((d_village < 1900) & (pente < 50) & (alea < 0.5)) | ((d_route < 200) & (alea < 0.0))
    occupation[champs & ~m_pnkb] = 4
    occupation[m_pnkb] = np.where(alea[m_pnkb] > -1.2, 1, 2)
    occupation[m_pnkb & (d_pnkb == 0) & (ndimage.distance_transform_edt(m_pnkb, sampling=RES) < 900)
               & (alea > 0.4)] = 2
    occupation[d_village < 150] = 6
    occupation[(d_riviere < 20) | m_lac] = 5

    # Végétation 2025 et 2015 (défrichements récents autour des villages).
    base = {1: 0.80, 2: 0.64, 3: 0.43, 4: 0.36, 5: -0.08, 6: 0.16}
    ndvi_2025 = np.vectorize(base.get, otypes=[float])(occupation) + bruit(rng, g.forme, 150, 0.04)
    ndvi_2025 = np.where((occupation == 4) & (pente > 35), ndvi_2025 - 0.12, ndvi_2025)
    defriche = (occupation == 4) & (d_village > 900) & (bruit(rng, g.forme, 700) > 0.9)
    ndvi_2015 = ndvi_2025 + np.where(defriche, 0.24, 0.0) + bruit(rng, g.forme, 300, 0.025)
    ndvi_2015 = np.where(occupation == 5, ndvi_2025, ndvi_2015)
    ndvi_2025, ndvi_2015 = np.clip(ndvi_2025, -0.2, 0.92), np.clip(ndvi_2015, -0.2, 0.92)

    couvert = np.clip((ndvi_2025 - 0.25) / 0.55 * 85, 0, 90)
    pluie = ndimage.gaussian_filter(1380 + 0.32 * (altitude - 1465), 4000 / RES) + bruit(rng, g.forme, 6000, 35)
    humidite = np.clip(ndimage.gaussian_filter(0.2 + 0.18 * ndvi_2025 + 0.00003 * (altitude - 1465), 3000 / RES), 0.1, 0.5)
    intensite_feux = 0.35 + 2.4 * ((occupation == 3) | (occupation == 4)) * np.exp(-d_village / 3000) \
        + 1.2 * ((d_pnkb > 0) & (d_pnkb < 2500))
    annees_brulees = np.minimum(rng.poisson(np.clip(ndimage.gaussian_filter(intensite_feux, 8), 0, None)), 11)
    annees_brulees = np.where(np.isin(occupation, (1, 5, 6)), 0, annees_brulees).astype(float)

    return {
        "dans_zone": dans_zone, "pnkb": m_pnkb, "occupation": occupation,
        "altitude": altitude, "pente": pente, "orientation": orientation,
        "ndvi_2015": ndvi_2015, "ndvi_2025": ndvi_2025, "changement_ndvi": ndvi_2025 - ndvi_2015,
        "couvert": couvert, "pluie": pluie, "humidite": humidite, "feux": annees_brulees,
        "d_pnkb": d_pnkb, "d_village": d_village, "d_route": d_route, "d_riviere": d_riviere,
        "d_champs": g.distance(occupation == 4),
    }


# --------------------------------------------------------------------------- écriture
def ecrire_raster(chemin: Path, valeurs, g, dans_zone, description, unite):
    zn.ecrire_raster(chemin, valeurs, g, dans_zone, description, unite, SOURCE, simulation="true")


ecrire_geojson = zn.ecrire_geojson


def main() -> None:
    parser = argparse.ArgumentParser(description="Génère des données simulées pour Kivu Smart.")
    parser.add_argument("--graine", type=int, default=2026, help="graine aléatoire (reproductibilité)")
    args = parser.parse_args()
    rng = np.random.default_rng(args.graine)
    g = Grille()
    print(f"Grille : {g.largeur} x {g.hauteur} pixels de {RES:.0f} m (UTM 35S)")

    zone, lac, pnkb, villages, routes, rivieres = construire_vecteurs(rng)
    r = construire_rasters(rng, g, zone, lac, pnkb, villages, routes, rivieres)
    masques, notes, pente_lissee = analyser(r)

    # ---- vecteurs de contexte
    v = SORTIE / "vecteurs"
    ecrire_geojson(v / "zone_etude.geojson", [{"id": "zone", "nom": "Secteur de Kalehe — zone d'étude", "geometry": zone}])
    ecrire_geojson(v / "pnkb.geojson", [{"id": "pnkb", "nom": "Parc national de Kahuzi-Biega (emprise indicative)",
                                         "type": "Aire protégée", "geometry": pnkb.intersection(zone)}])
    ecrire_geojson(v / "lac_kivu.geojson", [{"id": "lac", "nom": "Lac Kivu",
                                             "type": "Eau", "geometry": lac.intersection(zone)}])
    ecrire_geojson(v / "villages.geojson", [
        {"id": f"village-{i:02d}", "nom": f"Localité {i:02d}", "type": "Village",
         "geometry": wgs(utm(p).buffer(150))} for i, p in enumerate(villages, start=1)])
    ecrire_geojson(v / "routes.geojson", [
        {"id": f"route-{i:02d}", "nom": f"Route {i:02d}", "type": "Route",
         "geometry": ligne.intersection(zone)} for i, ligne in enumerate(routes, start=1)
        if not ligne.intersection(zone).is_empty])
    ecrire_geojson(v / "rivieres.geojson", [
        {"id": f"riviere-{i}", "nom": f"Rivière {i}", "type": "Cours d'eau",
         "geometry": ligne.intersection(zone)} for i, ligne in enumerate(rivieres, start=1)])

    # ---- zones de restauration
    zones = zn.zones_par_strategie(masques, notes, pente_lissee, g, r, surface_min_ha=8)
    for nom, entites in zones.items():
        ecrire_geojson(v / f"zones_{nom}.geojson", entites)
        print(f"  zones {nom:<15} : {len(entites):>3} polygone(s), "
              f"{sum(e['surface_ha'] for e in entites):,.0f} ha".replace(",", " "))

    # ---- rasters
    rr = SORTIE / "rasters"
    dz = r["dans_zone"]
    rasters = [
        ("altitude_m", r["altitude"], "Altitude", "m", False),
        ("pente_pct", r["pente"], "Pente", "%", False),
        ("orientation_deg", r["orientation"], "Orientation (0 = nord)", "degrés", False),
        ("ndvi_2015", r["ndvi_2015"], "NDVI 2015 (saison sèche)", "sans unité", False),
        ("ndvi_2025", r["ndvi_2025"], "NDVI 2025 (saison sèche)", "sans unité", False),
        ("changement_ndvi_2015_2025", r["changement_ndvi"], "Variation NDVI 2015-2025", "sans unité", False),
        ("couverture_arboree_pct", r["couvert"], "Couverture arborée", "%", False),
        ("pluie_mm_an", r["pluie"], "Précipitations annuelles", "mm/an", False),
        ("humidite_sol", r["humidite"], "Humidité du sol (zone racinaire)", "m3/m3", False),
        ("annees_brulees_2015_2025", r["feux"], "Fréquence des feux 2015-2025", "années", False),
        ("occupation_sol_2025", r["occupation"], "Occupation du sol 2025", "classe", True),
    ]
    for nom, valeurs, description, unite, _ in rasters:
        ecrire_raster(rr / f"{nom}.tif", valeurs, g, dz, description, unite)

    # ---- configuration du frontend (mode simulation)
    CONFIG.mkdir(parents=True, exist_ok=True)
    chemin = lambda p: p.relative_to(RACINE).as_posix()
    (CONFIG / "study_area.json").write_text(json.dumps({
        "name": "Secteur de Kalehe",
        "country": "République démocratique du Congo",
        "boundary_path": chemin(v / "zone_etude.geojson"),
        "boundary_source": "Jeu de démonstration (limite indicative, non officielle)",
        "boundary_source_crs": "EPSG:4326",
        "target_crs": SCR,
        "target_resolution_m": RES,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    couche = lambda nom, fichier, role: {
        "name": nom, "path": chemin(v / fichier), "role": role, "source": SOURCE,
        "satellite": False, "period_start": "2015-01-01", "period_end": "2025-12-31"}
    affichees = {"occupation_sol_2025": True}
    palettes = {"altitude_m": "altitude", "pente_pct": "pente", "ndvi_2015": "vegetation",
                "ndvi_2025": "vegetation", "changement_ndvi_2015_2025": "variation",
                "couverture_arboree_pct": "vegetation", "pluie_mm_an": "eau", "humidite_sol": "eau",
                "annees_brulees_2015_2025": "feux", "occupation_sol_2025": "categorique"}
    (CONFIG / "map_layers.json").write_text(json.dumps({
        "layers": [
            couche("Parc national de Kahuzi-Biega", "pnkb.geojson", "environmental"),
            couche("Lac Kivu", "lac_kivu.geojson", "environmental"),
            couche("Cours d'eau", "rivieres.geojson", "environmental"),
            couche("Routes", "routes.geojson", "environmental"),
            couche("Localités", "villages.geojson", "environmental"),
            couche("Zones prioritaires", "zones_prioritaire.geojson", "priority"),
            couche("Reboisement pur", "zones_reboisement.geojson", "forestry"),
            couche("Régénération naturelle assistée", "zones_rna.geojson", "rna"),
            couche("Agroforesterie", "zones_agroforesterie.geojson", "agroforestry"),
        ],
        "rasters": [
            {
                "id": nom, "name": description, "path": chemin(rr / f"{nom}.tif"), "source": SOURCE,
                "satellite": False, "product": "Jeu de démonstration", "categorical": categoriel,
                "unit": unite, "palette": palettes[nom], "show": affichees.get(nom, False),
                **({"class_colors": {str(k): c for k, (_, c) in CLASSES.items()},
                    "class_labels": {str(k): n for k, (n, _) in CLASSES.items()}} if categoriel else {}),
            }
            for nom, _, description, unite, categoriel in rasters
            if nom in palettes
        ],
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    (SORTIE / "LISEZMOI.txt").write_text(
        "DONNÉES SIMULÉES — générées par scripts/generer_donnees_simulees.py\n"
        "Aucune de ces couches ne provient de la NASA ni d'un relevé de terrain.\n"
        "Elles servent uniquement à tester l'affichage et la chaîne de traitement.\n"
        f"Graine aléatoire : {args.graine}\n", encoding="utf-8")
    print(f"Données écrites dans {SORTIE}")
    print(f"Configuration du frontend écrite dans {CONFIG}")
    print('Affichage : $env:KIVU_SIMULATION = "1"; streamlit run app.py')


if __name__ == "__main__":
    main()
