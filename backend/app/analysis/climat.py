"""Indicateurs climatiques et synthèse multi-modèles des scénarios CMIP6.

À partir des séries journalières moyennées sur la zone d'étude :

* indicateurs annuels utiles à la restauration (pluie, températures, extrêmes) ;
* anomalies par période future par rapport à la référence 1995-2014 (celle du GIEC AR6),
  calculées modèle par modèle puis résumées par la médiane et l'intervalle 10-90 % ;
* implications pour les stratégies RNA / agroforesterie / reboisement, déduites de règles
  explicites (et non d'un modèle « boîte noire »).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

REFERENCE = (1995, 2014)
PERIODES = {"2021-2040": (2021, 2040), "2041-2060": (2041, 2060), "2081-2100": (2081, 2100)}
GRADIENT_THERMIQUE_C_PAR_M = 0.0065  # baisse moyenne de température avec l'altitude
SEUIL_JOUR_SEC_MM = 1.0
SEUIL_PLUIE_FORTE_MM = 20.0
SEUIL_JOUR_CHAUD_C = 30.0

INDICATEURS = {
    "pluie_mm": ("Pluie annuelle", "mm", "relatif"),
    "tas_c": ("Température moyenne", "°C", "absolu"),
    "tasmax_c": ("Température maximale moyenne", "°C", "absolu"),
    "jours_chauds": (f"Jours > {SEUIL_JOUR_CHAUD_C:.0f} °C", "jours/an", "absolu"),
    "rx1day_mm": ("Pluie journalière maximale", "mm", "relatif"),
    "jours_pluie_forte": (f"Jours de pluie ≥ {SEUIL_PLUIE_FORTE_MM:.0f} mm", "jours/an", "absolu"),
    "cdd_jours": ("Plus longue période sèche", "jours", "absolu"),
    "pluie_saison_seche_mm": ("Pluie de juin à août", "mm", "relatif"),
}


def plus_longue_serie(masque: np.ndarray) -> int:
    """Longueur de la plus longue suite de ``True``."""
    meilleur = courant = 0
    for valeur in np.asarray(masque, dtype=bool):
        courant = courant + 1 if valeur else 0
        meilleur = max(meilleur, courant)
    return meilleur


def indicateurs_annuels(pr: np.ndarray | None, tas: np.ndarray | None, tasmax: np.ndarray | None,
                        mois: np.ndarray) -> dict[str, float]:
    """Indicateurs d'une année à partir des séries journalières de la zone."""
    sortie: dict[str, float] = {}
    if pr is not None and np.isfinite(pr).any():
        pr = np.nan_to_num(pr, nan=0.0)
        sortie.update(
            pluie_mm=float(pr.sum()),
            rx1day_mm=float(pr.max()),
            jours_pluie_forte=float((pr >= SEUIL_PLUIE_FORTE_MM).sum()),
            cdd_jours=float(plus_longue_serie(pr < SEUIL_JOUR_SEC_MM)),
            pluie_saison_seche_mm=float(pr[np.isin(mois[: pr.size], (6, 7, 8))].sum()),
        )
    if tas is not None and np.isfinite(tas).any():
        sortie["tas_c"] = float(np.nanmean(tas))
    if tasmax is not None and np.isfinite(tasmax).any():
        sortie["tasmax_c"] = float(np.nanmean(tasmax))
        sortie["jours_chauds"] = float((tasmax > SEUIL_JOUR_CHAUD_C).sum())
    return sortie


def climatologie_mensuelle(pr: np.ndarray | None, tas: np.ndarray | None, mois: np.ndarray) -> list[dict[str, Any]]:
    lignes = []
    for m in range(1, 13):
        jours = mois == m
        ligne: dict[str, Any] = {"mois": m}
        if pr is not None:
            ligne["pluie_mm"] = float(np.nansum(pr[jours[: pr.size]]))
        if tas is not None:
            ligne["tas_c"] = float(np.nanmean(tas[jours[: tas.size]]))
        lignes.append(ligne)
    return lignes


def _moyenne_periode(annuel: pd.DataFrame, debut: int, fin: int) -> pd.DataFrame:
    selection = annuel[(annuel["annee"] >= debut) & (annuel["annee"] <= fin)]
    colonnes = [c for c in INDICATEURS if c in selection.columns]
    return selection.groupby(["modele", "scenario"], as_index=False)[colonnes].mean()


def anomalies(annuel: pd.DataFrame) -> pd.DataFrame:
    """Anomalie de chaque indicateur, par modèle, scénario et période future.

    ``annuel`` : colonnes ``modele, scenario, annee`` + indicateurs ; la référence est
    lue dans le scénario ``historical`` du même modèle. Les indicateurs « relatifs »
    (pluie) sont exprimés en %, les autres en différence absolue.
    """
    reference = _moyenne_periode(annuel[annuel["scenario"] == "historical"], *REFERENCE)
    reference = reference.drop(columns="scenario").set_index("modele")
    lignes = []
    futur = annuel[annuel["scenario"] != "historical"]
    for periode, (debut, fin) in PERIODES.items():
        moyennes = _moyenne_periode(futur, debut, fin)
        for _, ligne in moyennes.iterrows():
            if ligne["modele"] not in reference.index:
                continue
            ref = reference.loc[ligne["modele"]]
            for indicateur, (_, _, mode) in INDICATEURS.items():
                if indicateur not in ligne or indicateur not in ref or pd.isna(ligne[indicateur]):
                    continue
                base, valeur = float(ref[indicateur]), float(ligne[indicateur])
                if mode == "relatif":
                    delta = (valeur - base) / base * 100 if base else np.nan
                else:
                    delta = valeur - base
                lignes.append({"modele": ligne["modele"], "scenario": ligne["scenario"], "periode": periode,
                               "indicateur": indicateur, "reference": base, "futur": valeur, "anomalie": delta,
                               "unite_anomalie": "%" if mode == "relatif" else INDICATEURS[indicateur][1]})
    return pd.DataFrame(lignes)


def synthese_ensemble(table: pd.DataFrame) -> pd.DataFrame:
    """Médiane, 10e et 90e centiles et accord de signe des modèles."""
    if table.empty:
        return pd.DataFrame(columns=["scenario", "periode", "indicateur", "mediane", "p10", "p90",
                                     "n_modeles", "accord_signe", "reference", "unite_anomalie"])

    def resume(groupe: pd.DataFrame) -> pd.Series:
        valeurs = groupe["anomalie"].dropna().to_numpy()
        mediane = float(np.median(valeurs)) if valeurs.size else np.nan
        accord = float(np.mean(np.sign(valeurs) == np.sign(mediane))) if valeurs.size and mediane else 0.0
        return pd.Series({
            "mediane": mediane,
            "p10": float(np.percentile(valeurs, 10)) if valeurs.size else np.nan,
            "p90": float(np.percentile(valeurs, 90)) if valeurs.size else np.nan,
            "n_modeles": int(valeurs.size),
            "accord_signe": accord,
            "reference": float(groupe["reference"].median()),
            "unite_anomalie": groupe["unite_anomalie"].iloc[0],
        })

    return (table.groupby(["scenario", "periode", "indicateur"])[["anomalie", "reference", "unite_anomalie"]]
            .apply(resume).reset_index())


def valeur(synthese: pd.DataFrame, scenario: str, periode: str, indicateur: str, champ: str = "mediane"):
    ligne = synthese[(synthese["scenario"] == scenario) & (synthese["periode"] == periode)
                     & (synthese["indicateur"] == indicateur)]
    return None if ligne.empty else float(ligne[champ].iloc[0])


def decalage_altitudinal_m(rechauffement_c: float) -> float:
    """Remontée des étages thermiques équivalente à un réchauffement donné."""
    return rechauffement_c / GRADIENT_THERMIQUE_C_PAR_M


def implications(synthese: pd.DataFrame, scenario: str, periode: str) -> list[dict[str, str]]:
    """Conséquences pour la restauration, déduites des anomalies médianes (règles explicites)."""
    messages: list[dict[str, str]] = []
    dt = valeur(synthese, scenario, periode, "tas_c")
    dpluie = valeur(synthese, scenario, periode, "pluie_mm")
    drx = valeur(synthese, scenario, periode, "rx1day_mm")
    dfortes = valeur(synthese, scenario, periode, "jours_pluie_forte")
    dcdd = valeur(synthese, scenario, periode, "cdd_jours")
    dsec = valeur(synthese, scenario, periode, "pluie_saison_seche_mm")
    accord_pluie = valeur(synthese, scenario, periode, "pluie_mm", "accord_signe")

    if dt is not None:
        messages.append({
            "theme": "Choix des essences", "strategies": "reboisement, agroforesterie",
            "texte": (f"Réchauffement médian de +{dt:.1f} °C : les étages climatiques remontent d'environ "
                      f"{decalage_altitudinal_m(dt):.0f} m. Choisir des essences adaptées à une altitude "
                      "plus basse que celle du site, et éviter celles en limite chaude de leur aire."),
        })
    if drx is not None or dfortes is not None:
        hausse = (drx or 0) > 0 or (dfortes or 0) > 0
        details = []
        if drx is not None:
            details.append(f"pluie journalière maximale {drx:+.0f} %")
        if dfortes is not None:
            details.append(f"{dfortes:+.1f} jour(s) de forte pluie par an")
        messages.append({
            "theme": "Érosion des versants", "strategies": "reboisement, zones prioritaires",
            "texte": (", ".join(details).capitalize() + ". "
                      + ("Des pluies plus intenses renforcent la priorité au couvert permanent sur les "
                         "pentes > 35 % et aux haies anti-érosives." if hausse else
                         "Pas d'intensification nette des pluies extrêmes dans la médiane des modèles.")),
        })
    if dcdd is not None or dsec is not None:
        details = []
        if dcdd is not None:
            details.append(f"plus longue période sèche {dcdd:+.1f} jour(s)")
        if dsec is not None:
            details.append(f"pluie de juin-août {dsec:+.0f} %")
        plus_sec = (dcdd or 0) > 0 or (dsec or 0) < 0
        messages.append({
            "theme": "Survie des plants", "strategies": "reboisement, agroforesterie",
            "texte": (", ".join(details).capitalize() + ". "
                      + ("Saison sèche plus marquée : planter dès le début des pluies (septembre-octobre), "
                         "paillage et essences tolérantes à la sécheresse." if plus_sec else
                         "La saison sèche ne s'allonge pas dans la médiane : calendrier de plantation inchangé.")),
        })
    if dt is not None and dcdd is not None:
        risque = dt >= 1.0 or dcdd > 0
        messages.append({
            "theme": "Risque de feux", "strategies": "rna, reboisement",
            "texte": ((f"Air plus chaud (+{dt:.1f} °C) et période sèche {dcdd:+.1f} jour(s) : la végétation "
                       "sèche plus vite. Pare-feu et surveillance en saison sèche autour des parcelles en RNA, "
                       "critère « feux répétés » à suivre chaque année.") if risque else
                      "Pas d'aggravation nette des conditions sèches : le critère « feux répétés » reste suffisant."),
        })
    if dpluie is not None:
        fiable = accord_pluie is not None and accord_pluie >= 0.8
        messages.append({
            "theme": "Pluie annuelle", "strategies": "toutes",
            "texte": (f"Variation médiane de {dpluie:+.0f} % "
                      + ("(au moins 80 % des modèles d'accord sur le sens)." if fiable else
                         "(les modèles ne s'accordent pas sur le sens : signal incertain).")
                      + " La RNA, qui s'appuie sur des souches déjà adaptées, reste la stratégie la moins "
                        "exposée à cette incertitude."),
        })
    return messages
