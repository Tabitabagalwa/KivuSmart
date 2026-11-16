"""Carte interactive des zones d'intervention."""

from __future__ import annotations

import json
from html import escape

import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from .. import map_view as mv
from ..carte import FONDS, construire_carte
from ..donnees import charger
from ..theme import ORDRE, STRATEGIES, entete, nombre, puce

CLE_SELECTION = "zone_selection"


@st.cache_data(show_spinner=False)
def _statistiques(feature_json: str, rasters_json: str) -> pd.DataFrame:
    statistiques, _ = mv._selected_raster_statistics(json.loads(feature_json), json.loads(rasters_json))
    return pd.DataFrame(statistiques)


def _filtres(donnees):
    barre = st.sidebar
    barre.markdown("#### Affichage de la carte")
    fond = barre.selectbox("Fond de carte", list(FONDS), index=0)
    strategies = barre.pills(
        "Interventions", options=list(ORDRE), selection_mode="multi", default=list(ORDRE),
        format_func=lambda c: f"{STRATEGIES[c].icone} {STRATEGIES[c].court}")
    aptitude_min = barre.slider("Aptitude minimale", 1.0, 5.0, 1.0, 0.1,
                                help="Masque les zones dont l'aptitude moyenne est inférieure.")
    noms_rasters = [r["name"] for r in donnees.rasters]
    rasters = barre.multiselect("Couches thématiques", noms_rasters,
                                default=[r["name"] for r in donnees.rasters if r.get("show")],
                                help="Relief, végétation, feux… affichés sous les zones.")
    opacite = barre.slider("Opacité des couches thématiques", 0.2, 1.0, 0.7, 0.05)
    reperes = barre.multiselect("Repères", [c["name"] for c in donnees.contexte],
                                default=[c["name"] for c in donnees.contexte])
    return fond, strategies, aptitude_min, [r for r in donnees.rasters if r["name"] in rasters], opacite, set(reperes)


def _fiche(zone: pd.Series, donnees, rasters) -> None:
    s = STRATEGIES[str(zone["strategie"])]
    lignes = [("Intervention", s.nom), ("Surface", f"{nombre(zone['surface_ha'], 1)} ha")]
    if pd.notna(zone["aptitude"]):
        lignes.append(("Aptitude", f"{nombre(zone['aptitude'], 1)} / 5"))
    if zone.get("priorite"):
        lignes.append(("Niveau de priorité", str(zone["priorite"]).capitalize()))
    for cle, libelle, unite, dec in (("pente", "Pente moyenne", " %", 0), ("ndvi", "NDVI 2025", "", 2),
                                     ("variation_ndvi", "Variation NDVI 2015-2025", "", 2),
                                     ("altitude", "Altitude moyenne", " m", 0)):
        if pd.notna(zone[cle]):
            texte = nombre(zone[cle], dec)
            lignes.append((libelle, ("+" if cle == "variation_ndvi" and zone[cle] > 0 else "") + texte + unite))
    tableau = "".join(f"<tr><td>{escape(a)}</td><td><b>{escape(b)}</b></td></tr>" for a, b in lignes)
    st.markdown(
        f'<div class="kv-fiche" style="--c:{s.couleur}">{puce(s, s.nom)}'
        f'<div style="font-size:1.05rem;font-weight:700;margin-top:.45rem;color:#111827">{escape(zone["nom"])}</div>'
        f'<table>{tableau}</table></div>', unsafe_allow_html=True)
    if zone.get("justification"):
        st.markdown(f"**Pourquoi cette zone ?** {zone['justification']}")
    rasters = [r for r in rasters if not r.get("categorical")]  # une moyenne de classes n'a pas de sens
    if rasters:
        feature = {"type": "Feature", "properties": {"_map_id": zone["map_id"]},
                   "geometry": json.loads(gpd_to_geojson(zone["geometry"]))}
        with st.expander("Valeurs des couches thématiques sur la zone"):
            stats = _statistiques(json.dumps(feature), json.dumps(rasters, sort_keys=True))
            if stats.empty:
                st.caption("Aucune valeur disponible.")
            else:
                st.dataframe(
                    stats.rename(columns={"layer": "Couche", "mean": "Moyenne", "min": "Min", "max": "Max",
                                          "valid_pixel_count": "Pixels"})[["Couche", "Moyenne", "Min", "Max", "Pixels"]],
                    hide_index=True, width="stretch",
                    column_config={c: st.column_config.NumberColumn(format="%.2f") for c in ("Moyenne", "Min", "Max")})


def gpd_to_geojson(geometrie) -> str:
    from shapely.geometry import mapping

    return json.dumps(mapping(geometrie))


def afficher() -> None:
    donnees = charger()
    fond, strategies, aptitude_min, rasters, opacite, reperes = _filtres(donnees)
    zones = donnees.zones
    visibles = zones[zones["strategie"].isin(strategies or [])
                     & (zones["aptitude"].isna() | (zones["aptitude"] >= aptitude_min))]

    entete("Carte des zones d'intervention",
           "Cliquez une zone pour afficher sa fiche. Les couches et le fond de carte se règlent dans le panneau de gauche.")
    resume = " ".join(
        puce(STRATEGIES[c], f"{STRATEGIES[c].court} · {len(g)} · {nombre(g['surface_ha'].sum())} ha")
        for c, g in ((c, visibles[visibles["strategie"] == c]) for c in ORDRE) if len(g))
    st.markdown(resume or "Aucune zone ne correspond aux filtres.", unsafe_allow_html=True)

    selection = st.session_state.get(CLE_SELECTION)
    if selection not in set(visibles["map_id"]):
        selection = None

    gauche, droite = st.columns([3.15, 1.25], gap="medium")
    with gauche:
        carte = construire_carte(donnees, visibles, fond=fond, rasters=rasters, opacite=opacite,
                                 contexte_visible=reperes, selection=selection)
        retour = st_folium(carte, height=680, use_container_width=True,
                           returned_objects=["last_active_drawing"], key="carte_principale")
        clic = ((retour or {}).get("last_active_drawing") or {}).get("properties", {}).get("map_id")
        if clic and clic != st.session_state.get("dernier_clic"):
            st.session_state["dernier_clic"] = clic
            st.session_state[CLE_SELECTION] = clic
            st.rerun()

    with droite:
        options = list(visibles["map_id"])
        libelles = {m: f"{STRATEGIES[str(s)].icone} {n}" for m, n, s in
                    zip(visibles["map_id"], visibles["nom"], visibles["strategie"])}
        choix = st.selectbox("Rechercher une zone", [None] + options,
                             index=(options.index(selection) + 1) if selection in options else 0,
                             format_func=lambda m: "— Choisir une zone —" if m is None else libelles[m])
        if choix != selection:
            st.session_state[CLE_SELECTION] = choix
            st.rerun()
        if selection is None:
            st.info("Cliquez une zone sur la carte, ou choisissez-la dans la liste, pour afficher sa fiche.")
            meilleures = visibles.dropna(subset=["aptitude"]).nlargest(5, "aptitude")
            if not meilleures.empty:
                st.markdown("**Zones les plus aptes**")
                for _, z in meilleures.iterrows():
                    st.markdown(f"{STRATEGIES[str(z['strategie'])].icone} {z['nom']} — "
                                f"**{nombre(z['aptitude'], 1)}/5**, {nombre(z['surface_ha'])} ha")
        else:
            _fiche(visibles[visibles["map_id"] == selection].iloc[0], donnees, rasters)
            if st.button("Effacer la sélection", width="stretch"):
                st.session_state[CLE_SELECTION] = None
                st.rerun()
