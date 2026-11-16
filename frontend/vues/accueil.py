"""Tableau de bord : vue d'ensemble pour ouvrir la présentation."""

from __future__ import annotations

import streamlit as st
from streamlit_folium import st_folium

from ..carte import construire_carte
from ..donnees import charger, synthese
from ..graphiques import barres_surface
from ..navigation import lien
from ..theme import ORDRE, STRATEGIES, carte_strategie, etape, hero, kpi, nombre


def afficher() -> None:
    donnees = charger()
    zones = donnees.zones
    tableau = synthese(zones).set_index("strategie")

    hero(
        "Restaurer les collines du Sud-Kivu, parcelle par parcelle",
        "Kivu Smart croise l'observation de la Terre (relief, végétation, climat, feux) et les données "
        "locales pour indiquer où intervenir et avec quelle méthode : régénération naturelle assistée, "
        "agroforesterie ou reboisement pur.",
        etiquette=f"{donnees.nom_zone} · NASA Space Apps Challenge 2026 — Field Shift",
    )

    restauration = zones[zones["strategie"].isin(["rna", "agroforesterie", "reboisement"])]
    prioritaires = zones[zones["strategie"] == "prioritaire"]
    surface_restauration = float(restauration["surface_ha"].sum())
    colonnes = st.columns(4)
    with colonnes[0]:
        kpi("Zone d'étude", f"{nombre(donnees.surface_zone_ha or 0)} ha",
            f"{nombre((donnees.surface_zone_ha or 0) / 100)} km²")
    with colonnes[1]:
        kpi("Zones d'intervention", f"{len(restauration)}", "délimitées automatiquement")
    with colonnes[2]:
        part = surface_restauration / donnees.surface_zone_ha * 100 if donnees.surface_zone_ha else 0
        kpi("Surface à restaurer", f"{nombre(surface_restauration)} ha", f"{nombre(part, 1)} % de la zone d'étude")
    with colonnes[3]:
        kpi("Zones prioritaires", f"{nombre(prioritaires['surface_ha'].sum())} ha",
            f"{len(prioritaires)} secteurs à traiter en premier")

    st.write("")
    colonnes = st.columns(4)
    for colonne, cle in zip(colonnes, ORDRE):
        s = STRATEGIES[cle]
        with colonne:
            if cle in tableau.index and tableau.loc[cle, "zones"] > 0:
                ligne = tableau.loc[cle]
                carte_strategie(s, f"{nombre(ligne['surface_ha'])} ha",
                                f"{int(ligne['zones'])} zones · aptitude {nombre(ligne['aptitude'], 1)}/5. {s.resume}")
            else:
                carte_strategie(s, "—", s.resume)

    st.write("")
    gauche, droite = st.columns([1.05, 1])
    with gauche:
        st.altair_chart(barres_surface(synthese(zones)), width="stretch")
        st.markdown("##### Parcours de la présentation")
        etapes = st.columns(2)
        with etapes[0]:
            etape(1, "Explorer la carte", "Zones d'intervention, relief, végétation et feux.")
            lien("carte", "Ouvrir la carte", "🗺️")
        with etapes[1]:
            etape(2, "Analyser les zones", "Classer, filtrer et exporter les zones.")
            lien("analyse", "Ouvrir l'analyse", "📊")
        etapes = st.columns(2)
        with etapes[0]:
            etape(3, "Comparer les stratégies", "Critères, bénéfices, risques et essences.")
            lien("strategies", "Voir la comparaison", "⚖️")
        with etapes[1]:
            etape(4, "Tester un terrain", "Le simulateur explique chaque décision.")
            lien("simulateur", "Ouvrir le simulateur", "🧪")
        etapes = st.columns(2)
        with etapes[0]:
            etape(5, "Anticiper le climat", "Scénarios CMIP6 de la NASA : chaleur, pluies, saison sèche.")
            lien("climat", "Voir les scénarios", "🌡️")
        with etapes[1]:
            etape(6, "Sources et limites", "Données utilisées, règles et validation de terrain.")
            lien("methodologie", "Voir la méthodologie", "📘")
    with droite:
        st.markdown("##### Aperçu de la zone d'étude")
        carte = construire_carte(donnees, zones, fond="Imagerie satellite", outils=False)
        st_folium(carte, height=470, use_container_width=True, returned_objects=[], key="apercu_accueil")
        st.caption("Violet : RNA · Ambre : agroforesterie · Vert foncé : reboisement pur · "
                   "Rouge hachuré : zones prioritaires.")
