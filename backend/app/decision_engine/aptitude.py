"""Règles d'aptitude spatiale du document « Facteurs de reboisement et API NASA ».

Utilisées pixel par pixel (tableaux numpy) pour produire les cartes de zones, et
valeur par valeur (scalaires) par le simulateur de l'interface : les deux donnent
donc toujours le même résultat.

* Critères déterminants = filtres (tout ou rien) ;
* facteurs complémentaires = notes de 1 (très faible) à 5 (très élevée),
  combinées par une moyenne pondérée (poids AHP) ;
* exclusions communes : eau et bâti ; RNA près / dans le PNKB, reboisement pur et
  agroforesterie hors PNKB.

Les poids du reboisement (pente 37,44 %, rivières 23,70 %) viennent du document
source ; les autres seuils et poids sont illustratifs et doivent être calibrés par
AHP avec les experts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

CLASSES_OCCUPATION = {
    1: "Forêt primaire",
    2: "Forêt secondaire",
    3: "Savane / prairie",
    4: "Mosaïque champs-jachère",
    5: "Eau",
    6: "Bâti",
}

SEUILS = {
    "rna": {"pente_min": 25.0, "pente_max": 35.0, "distance_aire_max_m": 3000.0, "feux_max": 6.0},
    "agroforesterie": {"distance_champs_max_m": 500.0, "distance_village_max_m": 2000.0, "pente_max": 25.0},
    "reboisement": {"pente_min": 35.0, "ndvi_max": 0.7},
    "prioritaire": {"pente_champs_min": 35.0, "perte_ndvi": -0.15},
}


@dataclass(frozen=True)
class Facteur:
    variable: str
    bornes: tuple[float, ...]
    notes: tuple[float, ...]
    poids: float
    libelle: str


FACTEURS: dict[str, tuple[Facteur, ...]] = {
    "rna": (
        Facteur("d_pnkb", (500, 1500, 3000), (5, 4, 3, 2), 0.35, "Proximité de l'aire protégée"),
        Facteur("ndvi", (0.3, 0.45, 0.6, 0.7), (1, 2, 3, 4, 5), 0.20, "Végétation résiduelle (NDVI)"),
        Facteur("couvert", (10, 25, 40, 60), (1, 2, 3, 4, 5), 0.15, "Couvert arboré"),
        Facteur("feux", (1, 3, 5), (5, 4, 2, 1), 0.15, "Faible fréquence des feux"),
        Facteur("d_village", (500, 1000, 2000, 3000), (1, 2, 3, 4, 5), 0.15, "Éloignement des villages"),
    ),
    "agroforesterie": (
        Facteur("d_champs", (100, 250, 500), (5, 4, 3, 2), 0.30, "Proximité des champs"),
        Facteur("d_village", (500, 1000, 1500, 2000), (5, 4, 3, 2, 1), 0.25, "Proximité des villages"),
        Facteur("pente", (8, 15, 20, 25), (5, 4, 3, 2, 1), 0.15, "Pente faible"),
        Facteur("d_route", (300, 800, 1500, 3000), (5, 4, 3, 2, 1), 0.15, "Accès par la route"),
        Facteur("d_riviere", (200, 500, 1000, 2000), (5, 4, 3, 2, 1), 0.15, "Accès à l'eau"),
    ),
    "reboisement": (
        Facteur("pente", (40, 50, 60), (2, 3, 4, 5), 0.3744, "Pente forte (risque d'érosion)"),
        Facteur("d_riviere", (100, 300, 600, 1000), (5, 4, 3, 2, 1), 0.2370, "Protection des berges"),
        Facteur("ndvi", (0.25, 0.35, 0.45, 0.55), (5, 4, 3, 2, 1), 0.1462, "Couvert dégradé (NDVI bas)"),
        Facteur("feux", (1, 3, 5), (5, 4, 2, 1), 0.1212, "Faible fréquence des feux"),
        Facteur("d_route", (500, 1500, 3000), (5, 4, 3, 2), 0.1212, "Accès pour la plantation"),
    ),
}

VARIABLES = ("pente", "occupation", "dans_pnkb", "d_pnkb", "d_champs", "d_village", "d_route",
             "d_riviere", "ndvi", "variation_ndvi", "couvert", "feux")


def classer(x: Any, bornes, notes) -> np.ndarray:
    """Note 1-5 selon les bornes (``notes`` a une valeur de plus que ``bornes``)."""
    return np.asarray(notes, dtype="float32")[np.digitize(x, bornes)]


def notes_facteurs(processus: str, v: dict[str, Any]) -> dict[str, np.ndarray]:
    return {f.libelle: classer(v[f.variable], f.bornes, f.notes) for f in FACTEURS[processus]}


def aptitude(processus: str, v: dict[str, Any]) -> np.ndarray:
    """Moyenne pondérée des notes 1-5 des facteurs complémentaires."""
    facteurs = FACTEURS[processus]
    total = sum(f.poids for f in facteurs)
    return sum(classer(v[f.variable], f.bornes, f.notes) * f.poids for f in facteurs) / total


def criteres(v: dict[str, Any]) -> dict[str, np.ndarray]:
    """Masques des trois processus (priorité : reboisement, puis RNA, puis agroforesterie)
    et des zones prioritaires (champs sur versant raide ou perte de végétation)."""
    s = SEUILS
    occ = np.asarray(v["occupation"])
    pnkb = np.asarray(v["dans_pnkb"], dtype=bool)
    pente = np.asarray(v["pente"], dtype=float)
    exclu = np.isin(occ, (5, 6)) | ~np.asarray(v.get("dans_zone", True), dtype=bool)

    reboisement = ((pente > s["reboisement"]["pente_min"]) & np.isin(occ, (2, 3)) & ~pnkb & ~exclu
                   & (np.asarray(v["ndvi"]) < s["reboisement"]["ndvi_max"]))
    rna = ((pente >= s["rna"]["pente_min"]) & (pente <= s["rna"]["pente_max"])
           & (np.asarray(v["d_pnkb"]) <= s["rna"]["distance_aire_max_m"]) & np.isin(occ, (2, 3))
           & ~exclu & ~reboisement & (np.asarray(v["feux"]) < s["rna"]["feux_max"]))
    agroforesterie = ((np.asarray(v["d_champs"]) <= s["agroforesterie"]["distance_champs_max_m"])
                      & (np.asarray(v["d_village"]) <= s["agroforesterie"]["distance_village_max_m"])
                      & (pente < s["agroforesterie"]["pente_max"]) & np.isin(occ, (3, 4))
                      & ~pnkb & ~exclu & ~reboisement & ~rna)
    prioritaire = (((occ == 4) & (pente > s["prioritaire"]["pente_champs_min"]))
                   | (np.asarray(v["variation_ndvi"]) < s["prioritaire"]["perte_ndvi"])) & ~exclu
    return {"reboisement": reboisement, "rna": rna, "agroforesterie": agroforesterie,
            "prioritaire": prioritaire}


def verifier(v: dict[str, Any]) -> dict[str, list[tuple[str, bool]]]:
    """Détail critère par critère (pour une seule valeur de chaque variable)."""
    s = SEUILS
    occ = int(v["occupation"])
    pente = float(v["pente"])
    pnkb = bool(v["dans_pnkb"])
    eau_bati = occ in (5, 6)
    libelle_occ = CLASSES_OCCUPATION.get(occ, str(occ))
    return {
        "reboisement": [
            (f"Pente > {s['reboisement']['pente_min']:.0f} % (actuelle : {pente:.0f} %)",
             pente > s["reboisement"]["pente_min"]),
            (f"Hors usage agricole : savane ou forêt dégradée ({libelle_occ})", occ in (2, 3)),
            ("Hors du PNKB", not pnkb),
            (f"Couvert non dense : NDVI < {s['reboisement']['ndvi_max']:.1f}",
             float(v["ndvi"]) < s["reboisement"]["ndvi_max"]),
            ("Ni eau ni bâti", not eau_bati),
        ],
        "rna": [
            (f"Pente entre {s['rna']['pente_min']:.0f} et {s['rna']['pente_max']:.0f} % (actuelle : {pente:.0f} %)",
             s["rna"]["pente_min"] <= pente <= s["rna"]["pente_max"]),
            (f"À ≤ {s['rna']['distance_aire_max_m'] / 1000:.0f} km de l'aire protégée",
             float(v["d_pnkb"]) <= s["rna"]["distance_aire_max_m"]),
            (f"Potentiel de recrû : savane ou forêt secondaire ({libelle_occ})", occ in (2, 3)),
            (f"Feux répétés < {s['rna']['feux_max']:.0f} ans sur 11", float(v["feux"]) < s["rna"]["feux_max"]),
            ("Ni eau ni bâti", not eau_bati),
        ],
        "agroforesterie": [
            (f"À ≤ {s['agroforesterie']['distance_champs_max_m']:.0f} m des champs",
             float(v["d_champs"]) <= s["agroforesterie"]["distance_champs_max_m"]),
            (f"À ≤ {s['agroforesterie']['distance_village_max_m'] / 1000:.0f} km d'un village",
             float(v["d_village"]) <= s["agroforesterie"]["distance_village_max_m"]),
            (f"Pente < {s['agroforesterie']['pente_max']:.0f} % (actuelle : {pente:.0f} %)",
             pente < s["agroforesterie"]["pente_max"]),
            (f"Terre agricole ou savane ({libelle_occ})", occ in (3, 4)),
            ("Hors du PNKB", not pnkb),
        ],
    }


def evaluer(v: dict[str, Any]) -> dict[str, Any]:
    """Évaluation complète d'un point : éligibilité, aptitude et recommandation."""
    masques = {k: bool(m) for k, m in criteres(v).items()}
    details = verifier(v)
    resultats = {}
    for processus in ("rna", "agroforesterie", "reboisement"):
        resultats[processus] = {
            "eligible": all(ok for _, ok in details[processus]),
            "retenu": masques[processus],
            "aptitude": float(aptitude(processus, v)),
            "criteres": details[processus],
            "facteurs": {k: float(n) for k, n in notes_facteurs(processus, v).items()},
        }
    retenu = next((p for p in ("reboisement", "rna", "agroforesterie") if masques[p]), None)
    return {"processus": resultats, "recommande": retenu, "prioritaire": masques["prioritaire"]}
