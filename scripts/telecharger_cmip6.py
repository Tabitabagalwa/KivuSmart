"""
Télécharge et résume les projections climatiques CMIP6 de la NASA (NEX-GDDP-CMIP6)
pour le territoire de Kalehe.

Étapes :
  1. limite du territoire (data/reel/vecteurs/limite_kalehe.geojson, sinon geoBoundaries) ;
  2. pour chaque modèle × scénario × variable × année : découpage NetCDF sur l'emprise
     (serveur THREDDS du NCCS, sans compte), mis en cache dans data/raw/cmip6/ ;
  3. moyenne journalière sur le territoire (cellules de 0,25° pondérées par leur surface
     dans Kalehe), indicateurs annuels et climatologie mensuelle ;
  4. anomalies 2021-2040 / 2041-2060 / 2081-2100 par rapport à 1995-2014, résumé multi-modèles.

Sorties (data/processed/climat/) :
  cmip6_annuel.csv, cmip6_mensuel.csv, cmip6_anomalies.csv, cmip6_synthese.csv, cmip6_meta.json

Utilisation (depuis la racine du projet) :
    python scripts/telecharger_cmip6.py                      # 6 modèles, SSP1-2.6/2-4.5/5-8.5
    python scripts/telecharger_cmip6.py --modeles MRI-ESM2-0 --scenarios ssp245
    python scripts/telecharger_cmip6.py --resumer-seulement  # recalcul depuis le cache
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from shapely.geometry import shape

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE))
from KivuSmart.backend.app.analysis import climat  # noqa: E402
from KivuSmart.backend.app.services.data_ingestion import sources_ouvertes as so  # noqa: E402
from KivuSmart.backend.app.services.nasa import cmip6  # noqa: E402
from KivuSmart.backend.app.services.nasa.client import NasaHttpClient  # noqa: E402
from KivuSmart.backend.app.services.nasa.errors import NasaApiError  # noqa: E402

CACHE = RACINE / "data" / "raw" / "cmip6"
SORTIE = RACINE / "data" / "processed" / "climat"
LIMITE = RACINE / "data" / "reel" / "vecteurs" / "limite_kalehe.geojson"
VARIABLES = ["pr", "tas"]           # ajouter "tasmax" avec --variables pr tas tasmax
HISTORIQUE = (1995, 2014)
FUTUR = (2021, 2100)


def limite_zone():
    if LIMITE.is_file():
        collection = json.loads(LIMITE.read_text(encoding="utf-8"))
        return shape(collection["features"][0]["geometry"])
    with so.client() as http:
        entite = so.limite_geoboundaries(http)
    LIMITE.parent.mkdir(parents=True, exist_ok=True)
    LIMITE.write_text(json.dumps({"type": "FeatureCollection", "features": [entite]}, ensure_ascii=False),
                      encoding="utf-8")
    return shape(entite["geometry"])


def chemin_cache(modele: str, scenario: str, variable: str, annee: int) -> Path:
    return CACHE / modele / scenario / f"{variable}_{annee}.npz"


def bornes(scenario: str) -> tuple[int, int]:
    return HISTORIQUE if scenario == "historical" else FUTUR


# ------------------------------------------------------------------ téléchargement
def inventaire(client: NasaHttpClient, modeles, scenarios) -> list[tuple]:
    taches = []
    for modele in modeles:
        for scenario in ("historical", *scenarios):
            try:
                disponibles = cmip6.membres(client, modele, scenario)
            except NasaApiError as exc:
                print(f"  {modele}/{scenario} : indisponible ({exc})")
                continue
            if not disponibles:
                print(f"  {modele}/{scenario} : aucun membre")
                continue
            membre = "r1i1p1f1" if "r1i1p1f1" in disponibles else disponibles[0]
            debut, fin = bornes(scenario)
            for variable in VARIABLES:
                try:
                    fichiers = cmip6.lister_fichiers(client, modele, scenario, membre, variable)
                except NasaApiError as exc:
                    print(f"  {modele}/{scenario}/{variable} : {exc}")
                    continue
                taches += [(modele, scenario, variable, annee, chemin) for annee, chemin in fichiers.items()
                           if debut <= annee <= fin and not chemin_cache(modele, scenario, variable, annee).is_file()]
    return taches


def telecharger(modeles, scenarios, emprise: cmip6.Emprise, travailleurs: int) -> int:
    with NasaHttpClient(cmip6.SERVICE, timeout_s=120, max_retries=4) as client:
        taches = inventaire(client, modeles, scenarios)
        total = len(taches)
        print(f"{total} extraction(s) à télécharger ({travailleurs} en parallèle).", flush=True)

        def une(tache) -> None:
            modele, scenario, variable, annee, chemin = tache
            grille = cmip6.lire(cmip6.telecharger(client, chemin, variable, emprise), variable, annee)
            destination = chemin_cache(modele, scenario, variable, annee)
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporaire = destination.with_name(destination.stem + ".part.npz")
            np.savez_compressed(temporaire, lat=grille.lat, lon=grille.lon,
                                valeurs=grille.valeurs.astype("float32"), source=np.array(chemin))
            temporaire.replace(destination)

        erreurs, t0 = [], time.time()
        with ThreadPoolExecutor(max_workers=travailleurs) as pool:
            futurs = {pool.submit(une, t): t for t in taches}
            for n, futur in enumerate(as_completed(futurs), start=1):
                try:
                    futur.result()
                except Exception as exc:  # une année manquante ne bloque pas le reste
                    erreurs.append((futurs[futur][:4], f"{type(exc).__name__}: {str(exc)[:160]}"))
                if n % 100 == 0 or n == total:
                    ecoule = time.time() - t0
                    print(f"  {n}/{total} ({ecoule / 60:.1f} min, reste ~{ecoule / n * (total - n) / 60:.1f} min)",
                          flush=True)
    for tache, message in erreurs[:15]:
        print("  échec :", tache, message)
    if erreurs:
        print(f"  {len(erreurs)} échec(s) : relancez le script pour les reprendre.")
    return len(erreurs)


# ------------------------------------------------------------------------ résumé
def lire_cache(chemin: Path):
    with np.load(chemin) as f:
        return f["lat"], f["lon"], f["valeurs"].astype("float64")


def resumer(modeles, scenarios, zone) -> None:
    lignes_annuelles, lignes_mensuelles, poids_info = [], [], {}
    poids_par_grille: dict[tuple, np.ndarray] = {}
    for modele in modeles:
        for scenario in ("historical", *scenarios):
            debut, fin = bornes(scenario)
            for annee in range(debut, fin + 1):
                series, n_jours = {}, None
                for variable in VARIABLES:
                    chemin = chemin_cache(modele, scenario, variable, annee)
                    if not chemin.is_file():
                        continue
                    lat, lon, valeurs = lire_cache(chemin)
                    cle = (tuple(np.round(lat, 4)), tuple(np.round(lon, 4)))
                    if cle not in poids_par_grille:
                        poids_par_grille[cle] = cmip6.poids_cellules(lat, lon, zone)
                    poids = poids_par_grille[cle]
                    poids_info = {"lat": lat.tolist(), "lon": lon.tolist(), "poids": np.round(poids, 3).tolist()}
                    series[variable] = cmip6.moyenne_zone(valeurs, poids)
                    n_jours = valeurs.shape[0]
                if not series:
                    continue
                mois = cmip6.mois_des_jours(n_jours)
                indicateurs = climat.indicateurs_annuels(series.get("pr"), series.get("tas"),
                                                         series.get("tasmax"), mois)
                lignes_annuelles.append({"modele": modele, "scenario": scenario, "annee": annee, **indicateurs})
                for ligne in climat.climatologie_mensuelle(series.get("pr"), series.get("tas"), mois):
                    lignes_mensuelles.append({"modele": modele, "scenario": scenario, "annee": annee, **ligne})
    if not lignes_annuelles:
        raise SystemExit("Aucune donnée en cache : lancez d'abord le téléchargement.")

    SORTIE.mkdir(parents=True, exist_ok=True)
    annuel = pd.DataFrame(lignes_annuelles).sort_values(["modele", "scenario", "annee"])
    annuel.round(3).to_csv(SORTIE / "cmip6_annuel.csv", index=False)
    mensuel = pd.DataFrame(lignes_mensuelles)
    periodes = {"1995-2014": climat.REFERENCE, **climat.PERIODES}
    blocs = []
    for nom, (debut, fin) in periodes.items():
        sel = mensuel[(mensuel["annee"] >= debut) & (mensuel["annee"] <= fin)]
        if nom == "1995-2014":
            sel = sel[sel["scenario"] == "historical"]
        bloc = sel.groupby(["modele", "scenario", "mois"], as_index=False)[["pluie_mm", "tas_c"]].mean()
        bloc["periode"] = nom
        blocs.append(bloc)
    pd.concat(blocs).round(3).to_csv(SORTIE / "cmip6_mensuel.csv", index=False)

    table = climat.anomalies(annuel)
    table.round(4).to_csv(SORTIE / "cmip6_anomalies.csv", index=False)
    synthese = climat.synthese_ensemble(table)
    synthese.round(4).to_csv(SORTIE / "cmip6_synthese.csv", index=False)

    complets = annuel.groupby(["modele", "scenario"]).size().to_dict()
    (SORTIE / "cmip6_meta.json").write_text(json.dumps({
        "source": cmip6.SOURCE, "doi": cmip6.DOI, "resolution": "0,25° (~27 km), journalier, corrigé des biais (BCSD)",
        "zone": "Territoire de Kalehe (moyenne pondérée des cellules couvrant la limite geoBoundaries)",
        "variables": {v: cmip6.VARIABLES[v] for v in VARIABLES},
        "modeles": sorted(annuel["modele"].unique().tolist()),
        "scenarios": {s: cmip6.SCENARIOS[s] for s in sorted(annuel["scenario"].unique())},
        "reference": f"{climat.REFERENCE[0]}-{climat.REFERENCE[1]}", "periodes": list(climat.PERIODES),
        "annees_par_simulation": {f"{m}/{s}": n for (m, s), n in complets.items()},
        "grille": poids_info,
        "genere_le": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Résumé écrit dans {SORTIE}")
    for scenario in scenarios:
        for periode in climat.PERIODES:
            dt = climat.valeur(synthese, scenario, periode, "tas_c")
            dp = climat.valeur(synthese, scenario, periode, "pluie_mm")
            if dt is not None:
                print(f"  {scenario} {periode} : ΔT médiane {dt:+.2f} °C, Δpluie {dp:+.1f} %")


def main() -> None:
    parser = argparse.ArgumentParser(description="Projections CMIP6 NASA NEX-GDDP pour Kalehe.")
    parser.add_argument("--modeles", nargs="+", default=list(cmip6.MODELES_DEFAUT))
    parser.add_argument("--scenarios", nargs="+", default=["ssp126", "ssp245", "ssp585"],
                        choices=[s for s in cmip6.SCENARIOS if s != "historical"])
    parser.add_argument("--variables", nargs="+", default=list(VARIABLES), choices=list(cmip6.VARIABLES))
    parser.add_argument("--travailleurs", type=int, default=10, help="requêtes simultanées (≤ 12)")
    parser.add_argument("--resumer-seulement", action="store_true", help="ne télécharge rien")
    args = parser.parse_args()
    VARIABLES[:] = args.variables

    zone = limite_zone()
    ouest, sud, est, nord = zone.bounds
    emprise = cmip6.Emprise(ouest, sud, est, nord)
    print(f"Zone : Kalehe, emprise {ouest:.2f}, {sud:.2f}, {est:.2f}, {nord:.2f}")
    if not args.resumer_seulement:
        telecharger(args.modeles, args.scenarios, emprise, max(1, min(args.travailleurs, 12)))
    resumer(args.modeles, args.scenarios, zone)


if __name__ == "__main__":
    main()
