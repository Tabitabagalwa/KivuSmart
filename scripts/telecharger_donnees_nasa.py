"""
Téléchargement des données NASA pour Kivu Smart (script autonome).

Couvre les facteurs du document « Facteurs de reboisement et API NASA » :

  1. NASA POWER  (aucun compte)        -> CSV      data/raw/power/
  2. NASA FIRMS  (clé MAP_KEY gratuite) -> CSV      data/raw/firms/
  3. Google Earth Engine (compte GEE)   -> GeoTIFF  data/raw/gee/
       altitude, pente (%), orientation, NDVI HLS (saison sèche), NDVI MODIS,
       couverture arborée, pluie annuelle, humidité du sol, stress hydrique,
       fréquence des feux, biomasse.

Installation :
    pip install earthengine-api requests pyproj rasterio geopandas
    earthengine authenticate

Utilisation (depuis la racine du projet) :
    python scripts/telecharger_donnees_nasa.py --plan        # liste, sans rien télécharger
    python scripts/telecharger_donnees_nasa.py               # tout télécharger
    python scripts/telecharger_donnees_nasa.py --seulement power firms
    python scripts/telecharger_donnees_nasa.py --finaliser   # après copie des fichiers venus de Drive

Non couvert (pas d'API) : couches RGC (villages, routes, cours d'eau, aires protégées,
limites) et carte d'occupation du sol LULC. Elles doivent être fournies à la main.

Modifiez la section CONFIGURATION avant de lancer.
"""

import argparse
import csv
import io
import json
import math
import os
import zipfile
from pathlib import Path

import requests

# =============================================================================
# CONFIGURATION
# =============================================================================
PROJET_GEE = "MON_PROJET_GCP"       # identifiant du projet Google Cloud Earth Engine

# Zone d'étude : soit un fichier vectoriel (Shapefile, GeoJSON, GPKG), soit une emprise.
ZONE_FICHIER = None                 # ex. "data/boundaries/zone_etude.shp"
ZONE_EMPRISE = (28.745, -2.392, 28.831, -2.311)  # ouest, sud, est, nord (parcelles pilotes de démonstration)

ANNEES_NDVI = [2015, 2020, 2025]    # années du document source
SAISON_SECHE = ("06-01", "08-31")   # juin à août
PERIODE_CLIMAT = (2015, 2025)       # pluie, feux, stress hydrique
PERIODE_SMAP = (2023, 2025)         # humidité du sol (saison sèche)
PERIODE_POWER = (2006, 2025)

CLE_FIRMS = os.getenv("FIRMS_MAP_KEY", "")   # https://firms.modaps.eosdis.nasa.gov/api/map_key/
SOURCE_FIRMS = "VIIRS_NOAA20_NRT"

DOSSIER = Path("data/raw")
SCR = "EPSG:32735"                  # UTM 35S
NODATA = -9999.0
LIMITE_DIRECTE_OCTETS = 48 * 1024 * 1024     # au-delà : export vers Google Drive
DOSSIER_DRIVE = "kivu_smart"


# =============================================================================
# ZONE ET GRILLE
# =============================================================================
def emprise():
    """Emprise (ouest, sud, est, nord) en degrés WGS 84."""
    if ZONE_FICHIER:
        import geopandas as gpd

        zone = gpd.read_file(ZONE_FICHIER)
        if zone.crs is None:
            raise SystemExit(f"{ZONE_FICHIER} n'a pas de système de coordonnées déclaré.")
        return tuple(float(v) for v in zone.to_crs(4326).total_bounds)
    return ZONE_EMPRISE


def grille_utm(echelle):
    """Grille UTM 35S alignée sur des multiples de l'échelle (identique d'une année à l'autre)."""
    from pyproj import Transformer

    ouest, sud, est, nord = emprise()
    t = Transformer.from_crs("EPSG:4326", SCR, always_xy=True)
    xs, ys = t.transform([ouest, est, ouest, est], [sud, sud, nord, nord])
    gauche = math.floor(min(xs) / echelle) * echelle
    droite = math.ceil(max(xs) / echelle) * echelle
    bas = math.floor(min(ys) / echelle) * echelle
    haut = math.ceil(max(ys) / echelle) * echelle
    largeur = max(1, int(round((droite - gauche) / echelle)))
    hauteur = max(1, int(round((haut - bas) / echelle)))
    return [echelle, 0, gauche, 0, -echelle, haut], largeur, hauteur


# =============================================================================
# 1. NASA POWER
# =============================================================================
PARAMETRES_POWER = ["T2M", "T2M_MAX", "T2M_MIN", "PRECTOTCORR", "RH2M", "ALLSKY_SFC_SW_DWN", "GWETROOT"]


def telecharger_power():
    ouest, sud, est, nord = emprise()
    lat, lon = (sud + nord) / 2, (ouest + est) / 2
    print(f"-> NASA POWER au centre de la zone ({lat:.4f}, {lon:.4f}), {PERIODE_POWER[0]}-{PERIODE_POWER[1]}")
    reponse = requests.get(
        "https://power.larc.nasa.gov/api/temporal/monthly/point",
        params={"parameters": ",".join(PARAMETRES_POWER), "community": "AG",
                "latitude": round(lat, 4), "longitude": round(lon, 4),
                "start": PERIODE_POWER[0], "end": PERIODE_POWER[1], "format": "JSON"},
        timeout=120)
    reponse.raise_for_status()
    donnees = reponse.json()["properties"]["parameter"]
    dossier = DOSSIER / "power"
    dossier.mkdir(parents=True, exist_ok=True)
    chemin = dossier / f"power_mensuel_{PERIODE_POWER[0]}_{PERIODE_POWER[1]}.csv"
    periodes = sorted(k for k in donnees[PARAMETRES_POWER[0]] if not k.endswith("13"))  # AAAA13 = annuel
    with chemin.open("w", newline="", encoding="utf-8") as fichier:
        ecrivain = csv.writer(fichier)
        ecrivain.writerow(["periode"] + PARAMETRES_POWER)
        for periode in periodes:
            valeurs = [donnees[p].get(periode) for p in PARAMETRES_POWER]
            ecrivain.writerow([periode] + ["" if v in (None, -999, -999.0) else v for v in valeurs])
    print(f"   OK : {chemin} (PRECTOTCORR en mm/jour ; grille ~50 km)")


# =============================================================================
# 2. NASA FIRMS (feux actifs des 5 derniers jours)
# =============================================================================
def telecharger_firms():
    if not CLE_FIRMS:
        print("-> FIRMS ignoré : définissez la variable d'environnement FIRMS_MAP_KEY.")
        return
    ouest, sud, est, nord = emprise()
    print(f"-> NASA FIRMS ({SOURCE_FIRMS}, 5 derniers jours)")
    url = (f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{CLE_FIRMS}/{SOURCE_FIRMS}/"
           f"{ouest},{sud},{est},{nord}/5")
    reponse = requests.get(url, timeout=120)
    reponse.raise_for_status()
    if not reponse.text.lstrip().lower().startswith("latitude") and "latitude" not in reponse.text[:200].lower():
        raise RuntimeError(f"Réponse FIRMS inattendue : {reponse.text[:200]}")
    dossier = DOSSIER / "firms"
    dossier.mkdir(parents=True, exist_ok=True)
    chemin = dossier / f"firms_{SOURCE_FIRMS}_5j.csv"
    chemin.write_text(reponse.text, encoding="utf-8")
    print(f"   OK : {chemin} ({max(0, len(reponse.text.splitlines()) - 1)} détection(s)). "
          "L'historique des feux vient de MCD64A1 (partie GEE).")


# =============================================================================
# 3. GOOGLE EARTH ENGINE
# =============================================================================
def _mnt(ee):
    return ee.Image("NASA/NASADEM_HGT/001").select("elevation")


def _pente_pct(ee):
    degres = ee.Terrain.slope(_mnt(ee))
    return degres.multiply(math.pi / 180).tan().multiply(100)       # % = tan(theta) x 100


def _orientation(ee):
    return ee.Terrain.aspect(_mnt(ee))                               # 0 = nord, sens horaire


def _ndvi_hls(annee):
    def construire(ee):
        def calcul(img):
            fmask = img.select("Fmask")
            propre = fmask.bitwiseAnd(0b00001110).eq(0)              # nuage, adjacent, ombre
            return img.normalizedDifference(["B5", "B4"]).updateMask(propre)

        return (ee.ImageCollection("NASA/HLS/HLSL30/v002")           # Landsat uniquement (pas Sentinel)
                .filterDate(f"{annee}-{SAISON_SECHE[0]}", f"{annee}-{SAISON_SECHE[1]}")
                .map(calcul).median())
    return construire


def _ndvi_modis(annee):
    def construire(ee):
        return (ee.ImageCollection("MODIS/061/MOD13Q1")
                .filterDate(f"{annee}-{SAISON_SECHE[0]}", f"{annee}-{SAISON_SECHE[1]}")
                .select("NDVI").mean().multiply(0.0001))
    return construire


def _couverture_arboree(ee):
    collection = ee.ImageCollection("MODIS/061/MOD44B").sort("system:time_start", False)
    return ee.Image(collection.first()).select("Percent_Tree_Cover")  # dernière année disponible


def _pluie(ee):
    return (ee.ImageCollection("NASA/GPM_L3/IMERG_MONTHLY_V07")
            .filterDate(f"{PERIODE_CLIMAT[0]}-01-01", f"{PERIODE_CLIMAT[1]}-12-31")
            .select("precipitation").mean().multiply(24 * 365.25))    # mm/h -> mm/an


def _humidite_sol(bande):
    def construire(ee):
        collection = ee.ImageCollection("NASA/SMAP/SPL4SMGP/008").select(bande)
        saisons = [collection.filterDate(f"{a}-{SAISON_SECHE[0]}", f"{a}-{SAISON_SECHE[1]}").mean()
                   for a in range(PERIODE_SMAP[0], PERIODE_SMAP[1] + 1)]
        return ee.ImageCollection(saisons).mean()
    return construire


def _stress_hydrique(ee):
    collection = (ee.ImageCollection("MODIS/061/MOD16A2GF")
                  .filterDate(f"{PERIODE_CLIMAT[0]}-01-01", f"{PERIODE_CLIMAT[1]}-12-31"))
    et = collection.select("ET").sum()
    pet = collection.select("PET").sum()
    return ee.Image(1).subtract(et.divide(pet))                      # 1 - ET/PET


def _frequence_feux(ee):
    collection = ee.ImageCollection("MODIS/061/MCD64A1").select("BurnDate")
    annees = [collection.filterDate(f"{a}-01-01", f"{a}-12-31").max().gt(0).unmask(0)
              for a in range(PERIODE_CLIMAT[0], PERIODE_CLIMAT[1] + 1)]
    return ee.ImageCollection(annees).sum()                          # nombre d'années brûlées


def _biomasse(ee):
    return ee.Image("LARSE/GEDI/GEDI04_B_002").select("MU")          # Mg/ha


def produits():
    """Liste (nom du fichier, résolution en m, fonction de construction de l'image)."""
    c0, c1 = PERIODE_CLIMAT
    liste = [
        ("nasadem_altitude", 30, _mnt),
        ("nasadem_pente_pct", 30, _pente_pct),
        ("nasadem_orientation", 30, _orientation),
    ]
    for annee in ANNEES_NDVI:
        liste.append((f"hls_ndvi_saison_seche_{annee}", 30, _ndvi_hls(annee)))
        liste.append((f"modis_ndvi_saison_seche_{annee}", 250, _ndvi_modis(annee)))
    liste += [
        ("mod44b_couverture_arboree_pct", 250, _couverture_arboree),
        (f"imerg_pluie_mm_an_{c0}_{c1}", 11000, _pluie),
        (f"smap_humidite_racinaire_{PERIODE_SMAP[0]}_{PERIODE_SMAP[1]}", 9000, _humidite_sol("sm_rootzone")),
        (f"smap_humidite_surface_{PERIODE_SMAP[0]}_{PERIODE_SMAP[1]}", 9000, _humidite_sol("sm_surface")),
        (f"mod16_stress_hydrique_{c0}_{c1}", 500, _stress_hydrique),
        (f"mcd64_annees_brulees_{c0}_{c1}", 500, _frequence_feux),
        ("gedi_biomasse_mg_ha", 1000, _biomasse),
    ]
    return liste


def _extraire_tiff(contenu):
    if contenu[:4] in (b"II*\x00", b"MM\x00*"):
        return contenu
    if contenu[:2] == b"PK":                                         # réponse zippée
        with zipfile.ZipFile(io.BytesIO(contenu)) as archive:
            tif = [n for n in archive.namelist() if n.lower().endswith(".tif")]
            return archive.read(tif[0])
    raise ValueError(contenu[:300].decode("utf-8", errors="replace"))


def finaliser(dossier=None):
    """Déclare -9999 comme « sans donnée » dans tous les GeoTIFF du dossier.

    À relancer (--finaliser) après avoir copié les fichiers récupérés dans Google Drive.
    """
    import rasterio

    dossier = Path(dossier or DOSSIER / "gee")
    for chemin in sorted(dossier.glob("*.tif")):
        with rasterio.open(chemin, "r+") as raster:
            if raster.nodata != NODATA:
                raster.nodata = NODATA
                print(f"   nodata déclaré : {chemin.name}")


def telecharger_gee():
    import ee

    try:
        ee.Initialize(project=PROJET_GEE)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=PROJET_GEE)

    dossier = DOSSIER / "gee"
    dossier.mkdir(parents=True, exist_ok=True)
    bilan = []
    for nom, echelle, construire in produits():
        transform, largeur, hauteur = grille_utm(echelle)
        print(f"-> {nom} ({largeur}x{hauteur} px à {echelle} m)")
        try:
            image = construire(ee).toFloat().unmask(NODATA, False)
            if largeur * hauteur * 4 > LIMITE_DIRECTE_OCTETS:
                tache = ee.batch.Export.image.toDrive(
                    image=image, description=nom, folder=DOSSIER_DRIVE, fileNamePrefix=nom,
                    crs=SCR, crsTransform=transform, dimensions=f"{largeur}x{hauteur}",
                    fileFormat="GeoTIFF", maxPixels=1e10)
                tache.start()
                print(f"   trop volumineux : export lancé vers Google Drive/{DOSSIER_DRIVE} (tâche {tache.id})")
                bilan.append({"fichier": nom, "statut": "drive", "tache": tache.id})
                continue
            url = image.getDownloadURL({"name": nom, "format": "GEO_TIFF", "crs": SCR,
                                        "crs_transform": transform, "dimensions": f"{largeur}x{hauteur}"})
            reponse = requests.get(url, timeout=600)
            reponse.raise_for_status()
            chemin = dossier / f"{nom}.tif"
            chemin.write_bytes(_extraire_tiff(reponse.content))
            print(f"   OK : {chemin}")
            bilan.append({"fichier": str(chemin), "statut": "ok", "resolution_m": echelle})
        except Exception as erreur:                                  # un échec n'arrête pas les autres
            print(f"   ÉCHEC : {erreur}")
            bilan.append({"fichier": nom, "statut": "échec", "erreur": str(erreur)})
    finaliser(dossier)
    (dossier / "bilan.json").write_text(json.dumps(bilan, indent=2, ensure_ascii=False), encoding="utf-8")
    en_drive = [b["fichier"] for b in bilan if b["statut"] == "drive"]
    if en_drive:
        print(f"\nÀ récupérer dans Google Drive/{DOSSIER_DRIVE} puis copier dans {dossier} : {en_drive}")
        print("Relancez ensuite : python scripts/telecharger_donnees_nasa.py --finaliser")


# =============================================================================
ETAPES = {"power": telecharger_power, "firms": telecharger_firms, "gee": telecharger_gee}


def main():
    analyseur = argparse.ArgumentParser(description="Téléchargement des données NASA pour Kivu Smart.")
    analyseur.add_argument("--plan", action="store_true", help="Lister les fichiers prévus, sans télécharger")
    analyseur.add_argument("--seulement", nargs="+", choices=sorted(ETAPES), help="Limiter à certaines sources")
    analyseur.add_argument("--finaliser", action="store_true",
                           help="Déclarer le nodata des GeoTIFF déjà présents (après copie depuis Drive)")
    options = analyseur.parse_args()

    if options.finaliser:
        finaliser()
        return
    if options.plan:
        print(f"Zone (ouest, sud, est, nord) : {emprise()}")
        for nom, echelle, _ in produits():
            _, largeur, hauteur = grille_utm(echelle)
            mode = "Drive" if largeur * hauteur * 4 > LIMITE_DIRECTE_OCTETS else "direct"
            print(f"  {nom:<45} {largeur:>6}x{hauteur:<6} px à {echelle:>5} m  ({mode})")
        print("  + NASA POWER (CSV) et NASA FIRMS (CSV, si FIRMS_MAP_KEY est définie)")
        return
    for nom in options.seulement or ["power", "firms", "gee"]:
        try:
            ETAPES[nom]()
        except Exception as erreur:
            print(f"ÉCHEC de l'étape {nom} : {erreur}")


if __name__ == "__main__":
    main()
