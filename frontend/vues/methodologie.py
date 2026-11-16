"""Méthodologie, sources de données et limites."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ..donnees import charger, climat, rapport_reel
from ..theme import entete, etape, nombre

CHAINE = [
    ("Zone d'étude", "Limite officielle du territoire de Kalehe (OCHA, via geoBoundaries)."),
    ("Acquisition", "Archives NASA ouvertes (Planetary Computer, NCCS THREDDS, POWER) et OpenStreetMap."),
    ("Harmonisation", "Reprojection et grille commune de 30 m en UTM 35S."),
    ("Facteurs dérivés", "Pente, NDVI de saison sèche, couvert arboré, fréquence des feux."),
    ("Proximités", "Distances aux villages, routes, cours d'eau, champs, aires protégées."),
    ("Notation", "Chaque facteur complémentaire est noté de 1 à 5."),
    ("Combinaison", "Critères déterminants en filtre, facteurs pondérés par AHP."),
    ("Restitution", "Cartes d'aptitude, zones prioritaires, fiches de zone et scénarios climatiques."),
]

SOURCES = [
    ("Altitude, pente", "NASADEM_HGT v001", "NASA JPL / LP DAAC", "30 m"),
    ("Végétation (NDVI 2015 et 2025)", "Landsat 8-9 OLI Collection 2 niveau 2", "NASA / USGS", "30 m"),
    ("Couvert arboré et pertes", "Global Forest Change (Landsat)", "Hansen / UMD / Google / USGS / NASA", "30 m"),
    ("Feux (2015-2025)", "MCD64A1 v061", "NASA LP DAAC", "500 m"),
    ("Climat observé", "NASA POWER", "NASA LaRC", "≈ 50 km"),
    ("Scénarios climatiques", "NEX-GDDP-CMIP6 (SSP1-2.6, 2-4.5, 5-8.5)", "NASA NCCS / Ames", "≈ 27 km"),
    ("Limite de Kalehe", "Référentiel géographique commun", "OCHA RDC via geoBoundaries", "Vecteur"),
    ("Villages, routes, cours d'eau, champs, PNKB", "OpenStreetMap", "Contributeurs OSM (ODbL)", "Vecteur"),
    ("Occupation du sol", "Classification par règles (couvert, NDVI, OSM)", "Kivu Smart", "30 m"),
]


def afficher() -> None:
    donnees = charger()
    entete("Méthodologie et données", "De l'image satellite à la fiche de zone : la chaîne de traitement.")

    for debut in (0, 4):
        colonnes = st.columns(4)
        for i, colonne in enumerate(colonnes):
            titre, texte = CHAINE[debut + i]
            with colonne:
                etape(debut + i + 1, titre, texte)
        st.write("")

    gauche, droite = st.columns([1.5, 1], gap="large")
    with gauche:
        st.markdown("##### Sources de données")
        st.dataframe(pd.DataFrame(SOURCES, columns=["Facteur", "Produit", "Fournisseur", "Résolution"]),
                     hide_index=True, width="stretch")
        if donnees.demo:
            st.caption("Sources de la chaîne réelle (scripts/preparer_donnees_reelles.py). "
                       "Le jeu actuellement affiché est un jeu de démonstration.")
        st.markdown("##### Règles de décision")
        st.markdown(
            "- **RNA** : pente de 25 à 35 %, à moins de 3 km d'une aire protégée, savane ou forêt secondaire.\n"
            "- **Agroforesterie** : à moins de 500 m des champs et 2 km d'un village, pente < 25 %, hors PNKB.\n"
            "- **Reboisement pur** : pente > 35 %, hors usage agricole et hors PNKB.\n"
            "- **Zones prioritaires** : champs sur pente > 35 % ou perte de NDVI > 0,15 depuis 2015.\n"
            "- **Exclusions** : surfaces en eau et zones bâties.")
    with droite:
        st.markdown("##### Limites des données & validation locale")
        st.markdown(
            '<div class="kv-limites">La résolution varie de 30 m (Landsat, NASADEM, Hansen) à 500 m (feux MODIS) '
            "et 27 à 50 km (scénarios CMIP6, POWER) : les données grossières décrivent des tendances régionales. "
            "L'occupation du sol est déduite par règles et OpenStreetMap est incomplet en milieu rural. "
            "Les seuils et pondérations doivent être calibrés avec les experts. "
            "<b>Toute recommandation doit être validée sur le terrain, avec les agriculteurs et les agronomes "
            "locaux, avant plantation.</b></div>", unsafe_allow_html=True)
        st.markdown("##### Jeu de données affiché")
        st.markdown(f"- **Zone** : {donnees.nom_zone}"
                    + (f" ({nombre(donnees.surface_zone_ha / 100)} km²)" if donnees.surface_zone_ha else "") + "\n"
                    f"- **Limite** : {donnees.source_limite}\n"
                    f"- **Zones d'intervention** : {len(donnees.zones)}\n"
                    f"- **Couches thématiques** : {len(donnees.rasters)}\n"
                    f"- **Source des zones** : {donnees.zones['source'].iloc[0] if len(donnees.zones) else '—'}")
        rapport = {} if donnees.demo else rapport_reel()
        if rapport:
            scenes = rapport.get("landsat_scenes", {})
            valides = rapport.get("ndvi_pixels_valides_pct", {})
            with st.expander("Contrôle qualité des données réelles"):
                st.markdown(
                    f"- **Généré le** : {rapport.get('genere_le', '—')}, grille de {rapport.get('resolution_m', 30):.0f} m\n"
                    + "".join(f"- **NDVI {e}** : {len(scenes.get(e, []))} scène(s) Landsat, "
                              f"{nombre(valides.get(e, 0), 0)} % de pixels sans nuage\n" for e in scenes)
                    + f"- **Couvert forestier** : Hansen {rapport.get('hansen_version', '—')}\n"
                    f"- **Feux MODIS** : {len(rapport.get('modis_annees', []))} année(s)\n"
                    f"- **OpenStreetMap** : {rapport.get('osm', {}).get('localites', 0)} localités, "
                    f"{rapport.get('osm', {}).get('champs_cartographies', 0)} champs cartographiés")
                for limite in rapport.get("limites", []):
                    st.caption("• " + limite)
        c = climat()
        if c is not None:
            st.markdown(f"- **Scénarios climatiques** : {len(c.meta.get('modeles', []))} modèle(s) CMIP6, "
                        f"{len(c.meta.get('scenarios', {})) - 1} scénario(s), généré le {c.meta.get('genere_le', '—')}")
        if donnees.problemes:
            with st.expander(f"Couches non chargées ({len(donnees.problemes)})"):
                for probleme in donnees.problemes:
                    st.caption(probleme)
