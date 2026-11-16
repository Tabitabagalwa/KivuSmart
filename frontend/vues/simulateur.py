"""Simulateur : évaluer un terrain et voir pourquoi une stratégie est retenue."""

from __future__ import annotations

from html import escape

import pandas as pd
import streamlit as st

from KivuSmart.backend.app.decision_engine.aptitude import CLASSES_OCCUPATION, evaluer

from ..graphiques import barres_facteurs
from ..theme import BASELINE, STRATEGIES, entete, nombre

EXEMPLES = {
    "Lisière du parc, recrû forestier": dict(
        pente=30, occupation=2, dans_pnkb=False, d_pnkb=800, d_champs=1200, d_village=2600, d_route=2000,
        d_riviere=700, ndvi=0.58, variation_ndvi=0.0, couvert=45, feux=1),
    "Versant très pentu et dégradé": dict(
        pente=48, occupation=3, dans_pnkb=False, d_pnkb=6000, d_champs=900, d_village=2500, d_route=1200,
        d_riviere=250, ndvi=0.32, variation_ndvi=-0.05, couvert=10, feux=2),
    "Terroir agricole près d'un village": dict(
        pente=12, occupation=4, dans_pnkb=False, d_pnkb=7000, d_champs=0, d_village=600, d_route=300,
        d_riviere=400, ndvi=0.36, variation_ndvi=-0.03, couvert=8, feux=2),
    "Champ sur versant raide, défriché": dict(
        pente=42, occupation=4, dans_pnkb=False, d_pnkb=5000, d_champs=0, d_village=1500, d_route=900,
        d_riviere=500, ndvi=0.24, variation_ndvi=-0.22, couvert=4, feux=3),
}
PREFIXE = "sim_"


def _charger_exemple() -> None:
    for cle, valeur in EXEMPLES[st.session_state["sim_exemple"]].items():
        st.session_state[PREFIXE + cle] = valeur


def _saisie() -> dict:
    if PREFIXE + "pente" not in st.session_state:
        st.session_state["sim_exemple"] = next(iter(EXEMPLES))
        _charger_exemple()
    st.selectbox("Partir d'un exemple", list(EXEMPLES), key="sim_exemple", on_change=_charger_exemple)
    with st.container(border=True):
        st.markdown("**Terrain et végétation**")
        st.slider("Pente (%)", 0, 80, key=PREFIXE + "pente")
        st.selectbox("Occupation du sol", list(CLASSES_OCCUPATION), key=PREFIXE + "occupation",
                     format_func=CLASSES_OCCUPATION.get)
        st.slider("NDVI 2025 (végétation)", -0.1, 0.9, step=0.01, key=PREFIXE + "ndvi")
        st.slider("Variation du NDVI 2015-2025", -0.4, 0.3, step=0.01, key=PREFIXE + "variation_ndvi")
        st.slider("Couverture arborée (%)", 0, 90, key=PREFIXE + "couvert")
        st.slider("Années brûlées sur 2015-2025", 0, 11, key=PREFIXE + "feux")
    with st.container(border=True):
        st.markdown("**Distances et contexte**")
        st.toggle("Situé dans le PNKB", key=PREFIXE + "dans_pnkb")
        st.slider("Distance à l'aire protégée (m)", 0, 10000, step=100, key=PREFIXE + "d_pnkb")
        st.slider("Distance aux champs (m)", 0, 3000, step=50, key=PREFIXE + "d_champs")
        st.slider("Distance au village le plus proche (m)", 0, 6000, step=100, key=PREFIXE + "d_village")
        st.slider("Distance à une route (m)", 0, 5000, step=100, key=PREFIXE + "d_route")
        st.slider("Distance à un cours d'eau (m)", 0, 3000, step=50, key=PREFIXE + "d_riviere")
    valeurs = {cle: st.session_state[PREFIXE + cle] for cle in EXEMPLES[next(iter(EXEMPLES))]}
    if valeurs["dans_pnkb"]:
        valeurs["d_pnkb"] = 0
    return valeurs


def _bandeau(resultat: dict) -> None:
    cle = resultat["recommande"]
    s = STRATEGIES[cle] if cle else BASELINE
    if cle:
        titre = f"{s.icone} {s.nom}"
        texte = (f"Tous les critères déterminants sont remplis. Aptitude "
                 f"{nombre(resultat['processus'][cle]['aptitude'], 1)} / 5.")
    else:
        titre = "Aucune intervention ne remplit les critères"
        texte = "Maintenir la pratique actuelle comme référence et vérifier la situation sur le terrain."
    alerte = ""
    if resultat["prioritaire"]:
        alerte = (f'<div style="margin-top:.5rem;color:#7f1d1d;font-weight:600">'
                  f'{STRATEGIES["prioritaire"].icone} Zone prioritaire : champ sur versant raide ou perte de '
                  f'végétation récente — risque d\'érosion élevé.</div>')
    st.markdown(
        f'<div class="kv-fiche" style="--c:{s.couleur};padding:16px 18px">'
        f'<div style="font-size:.75rem;color:#6b7280;text-transform:uppercase;letter-spacing:.05em">Recommandation</div>'
        f'<div style="font-size:1.35rem;font-weight:700;color:#111827">{escape(titre)}</div>'
        f'<div style="color:#374151">{escape(texte)}</div>{alerte}</div>', unsafe_allow_html=True)


def afficher() -> None:
    entete("Simulateur de décision",
           "Réglez les caractéristiques d'un terrain : la plateforme applique les mêmes règles que pour la carte "
           "et explique sa décision critère par critère.")
    gauche, droite = st.columns([1, 1.9], gap="large")
    with gauche:
        valeurs = _saisie()
    resultat = evaluer(valeurs)
    with droite:
        _bandeau(resultat)
        st.write("")
        colonnes = st.columns(3)
        for colonne, cle in zip(colonnes, ["rna", "agroforesterie", "reboisement"]):
            s, r = STRATEGIES[cle], resultat["processus"][cle]
            if r["retenu"]:
                statut = "✅ Retenue"
            elif r["eligible"]:
                statut = "☑️ Éligible (une autre stratégie prime)"
            else:
                statut = "❌ Critères non remplis"
            with colonne, st.container(border=True):
                st.markdown(f'<div style="border-top:4px solid {s.couleur};margin:-4px 0 4px;padding-top:6px;'
                            f'font-weight:700;color:#111827">{s.icone} {escape(s.court)}</div>'
                            f'<div style="font-size:.85rem;color:#374151">{statut}</div>', unsafe_allow_html=True)
                st.progress(min(1.0, r["aptitude"] / 5), text=f"Aptitude {nombre(r['aptitude'], 1)} / 5")
                st.markdown("\n".join(f"- {'✔️' if ok else '✖️'} {texte}" for texte, ok in r["criteres"]))

        st.markdown("##### Détail des notes (1 à 5) par facteur")
        choix = resultat["recommande"] or "reboisement"
        onglets = st.tabs([f"{STRATEGIES[c].icone} {STRATEGIES[c].court}" for c in ("rna", "agroforesterie", "reboisement")])
        for onglet, cle in zip(onglets, ("rna", "agroforesterie", "reboisement")):
            with onglet:
                facteurs = pd.DataFrame([{"facteur": f, "valeur": n}
                                         for f, n in resultat["processus"][cle]["facteurs"].items()])
                st.altair_chart(barres_facteurs(facteurs, STRATEGIES[cle].couleur,
                                                f"{STRATEGIES[cle].nom}{' (recommandée)' if cle == choix and resultat['recommande'] else ''}",
                                                titre_axe="Note (1 à 5)", domaine=[0, 5]),
                                width="stretch")
        st.caption("Les critères déterminants filtrent ; les notes des facteurs complémentaires, pondérées "
                   "par AHP, donnent l'aptitude. Une recommandation se valide toujours sur le terrain.")
