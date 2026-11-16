"""Comparaison de la pratique actuelle et des trois stratégies de restauration."""

from __future__ import annotations

from html import escape

import pandas as pd
import streamlit as st

from KivuSmart.backend.app.decision_engine.aptitude import FACTEURS, SEUILS
from KivuSmart.backend.engine import SCENARIO_DETAILS

from ..donnees import charger, referentiel, synthese
from ..graphiques import barres_facteurs
from ..theme import BASELINE, STRATEGIES, entete, nombre

DETERMINANTS = {
    "rna": [f"Pente entre {SEUILS['rna']['pente_min']:.0f} et {SEUILS['rna']['pente_max']:.0f} %",
            f"À ≤ {SEUILS['rna']['distance_aire_max_m'] / 1000:.0f} km d'une aire protégée (PNKB)",
            "Savane ou forêt secondaire (potentiel de recrû)"],
    "agroforesterie": [f"À ≤ {SEUILS['agroforesterie']['distance_champs_max_m']:.0f} m des champs",
                       f"À ≤ {SEUILS['agroforesterie']['distance_village_max_m'] / 1000:.0f} km d'un village",
                       f"Pente < {SEUILS['agroforesterie']['pente_max']:.0f} %, hors PNKB"],
    "reboisement": [f"Pente > {SEUILS['reboisement']['pente_min']:.0f} %",
                    "Hors usage agricole (savane, forêt dégradée)", "Hors PNKB"],
    "baseline": ["Culture annuelle continue, y compris sur pente", "NDVI bas en saison sèche",
                 "Peu ou pas de couvert arboré"],
}
REFERENTIEL_CLE = {"rna": "rna", "agroforesterie": "agroforesterie", "reboisement": "reboisement"}

MATRICE = [
    ("Pente", "● 25–35 %", "○ < 25 %", "● > 35 %"),
    ("Proximité des aires protégées", "●", "Hors PNKB", "Hors PNKB"),
    ("Proximité des champs", "—", "●", "—"),
    ("Proximité des villages", "○ (pression)", "●", "○"),
    ("Proximité des routes", "—", "○", "○"),
    ("Proximité des cours d'eau", "—", "○", "○"),
    ("Occupation du sol", "○", "○", "○"),
    ("NDVI / EVI (végétation)", "○", "○", "○"),
    ("Couverture arborée", "○", "—", "—"),
    ("Humidité du sol, précipitations", "○", "○", "○"),
    ("Évapotranspiration", "—", "○", "○"),
    ("Fréquence des feux", "✕ contrainte", "○", "✕ contrainte"),
    ("Altitude, orientation", "—", "○", "○ (altitude)"),
]


def _liste(titre: str, elements: list[str]) -> str:
    puces = "".join(f"<li>{escape(e)}</li>" for e in elements)
    return f'<div style="font-weight:600;margin-top:.6rem;color:#111827">{escape(titre)}</div><ul style="margin:.2rem 0 0 1rem;padding:0">{puces}</ul>'


def _colonne(cle: str, surfaces: pd.DataFrame, essences: list[dict]) -> None:
    s = BASELINE if cle == "baseline" else STRATEGIES[cle]
    details = SCENARIO_DETAILS[cle]
    with st.container(border=True):
        st.markdown(
            f'<div style="border-top:4px solid {s.couleur};margin:-4px 0 6px;padding-top:8px">'
            f'<div style="font-size:1.08rem;font-weight:700;color:#111827">{s.icone} {escape(s.nom)}</div>'
            f'<div style="color:#4b5563;font-size:.88rem;margin-top:.2rem">{escape(details["summary"])}</div></div>',
            unsafe_allow_html=True)
        if cle in surfaces.index:
            ligne = surfaces.loc[cle]
            st.metric("Surface identifiée", f"{nombre(ligne['surface_ha'])} ha", f"{int(ligne['zones'])} zones",
                      delta_color="off")
        html = _liste("Critères déterminants", DETERMINANTS[cle])
        if cle in FACTEURS:
            html += _liste("Facteurs complémentaires (poids)",
                           [f"{f.libelle} — {f.poids * 100:.0f} %" for f in FACTEURS[cle]])
        html += _liste("Bénéfices attendus", details["benefits"])
        html += _liste("Risques et vigilance", details["risks"])
        noms = [e["name"] for e in essences if REFERENTIEL_CLE.get(cle) in e.get("recommended_for", [])]
        if noms:
            html += _liste("Essences proposées", noms)
        st.markdown(f'<div style="font-size:.86rem;color:#374151">{html}</div>', unsafe_allow_html=True)


def afficher() -> None:
    donnees = charger()
    entete("Comparer les stratégies",
           "La pratique actuelle sert de référence ; chaque stratégie a ses critères, bénéfices et risques.")
    surfaces = synthese(donnees.zones).set_index("strategie")
    essences = referentiel().get("species", [])

    colonnes = st.columns(4, gap="small")
    for colonne, cle in zip(colonnes, ["baseline", "rna", "agroforesterie", "reboisement"]):
        with colonne:
            _colonne(cle, surfaces, essences)

    st.markdown("##### Matrice des facteurs")
    st.caption("● critère déterminant (filtre)  ·  ○ facteur complémentaire (noté de 1 à 5)  ·  ✕ contrainte")
    st.dataframe(pd.DataFrame(MATRICE, columns=["Facteur", "RNA", "Agroforesterie", "Reboisement pur"]),
                 hide_index=True, width="stretch")

    st.markdown("##### Pondération des facteurs complémentaires")
    colonnes = st.columns(3, gap="large")
    for colonne, cle in zip(colonnes, ["rna", "agroforesterie", "reboisement"]):
        facteurs = pd.DataFrame([{"facteur": f.libelle, "valeur": f.poids * 100} for f in FACTEURS[cle]])
        with colonne:
            st.altair_chart(barres_facteurs(facteurs, STRATEGIES[cle].couleur, STRATEGIES[cle].nom,
                                            domaine=[0, 45]), width="stretch")
    st.caption("Poids issus de l'analyse multicritère AHP. Pour le reboisement pur, la pente (37,4 %) et la "
               "distance aux cours d'eau (23,7 %) reprennent le document source ; les autres poids sont à "
               "calibrer avec les experts.")
    _climat_futur()


def _climat_futur() -> None:
    """Conséquences des scénarios CMIP6 pour chaque stratégie (si les projections sont disponibles)."""
    from KivuSmart.backend.app.analysis import climat as cl

    from ..donnees import climat
    from ..navigation import lien

    c = climat()
    if c is None or "ssp245" not in set(c.synthese["scenario"]):
        return
    st.markdown("##### Face au climat futur (SSP2-4.5, 2041-2060)")
    messages = cl.implications(c.synthese, "ssp245", "2041-2060")
    colonnes = st.columns(3, gap="large")
    for colonne, cle in zip(colonnes, ["rna", "agroforesterie", "reboisement"]):
        s = STRATEGIES[cle]
        pertinents = [m for m in messages if cle in m["strategies"] or m["strategies"] == "toutes"]
        with colonne:
            html = "".join(f"<li><b>{escape(m['theme'])}</b> — {escape(m['texte'])}</li>" for m in pertinents)
            st.markdown(f'<div class="kv-carte" style="--c:{s.couleur}"><div class="tete">{s.icone} '
                        f'{escape(s.nom)}</div><ul class="texte" style="margin:.3rem 0 0 1rem;padding:0">{html}'
                        "</ul></div>", unsafe_allow_html=True)
    lien("climat", "Explorer les scénarios climatiques", "🌡️")
