"""Analyse et classement des zones d'intervention."""

from __future__ import annotations

import streamlit as st

from ..donnees import charger, synthese
from ..graphiques import barres_surface, nuage_aptitude, nuage_pente_ndvi
from ..navigation import aller
from ..theme import ORDRE, STRATEGIES, entete, kpi, nombre

NIVEAUX = ["très élevée", "élevée", "moyenne", "faible"]


def afficher() -> None:
    donnees = charger()
    zones = donnees.zones
    entete("Analyse des zones", "Filtrez, comparez et exportez les zones d'intervention identifiées.")

    filtres = st.columns([1.6, 1.4, 1, 1])
    strategies = filtres[0].multiselect("Interventions", list(ORDRE), default=list(ORDRE),
                                        format_func=lambda c: f"{STRATEGIES[c].icone} {STRATEGIES[c].nom}")
    niveaux_presents = [n for n in NIVEAUX if n in set(zones["priorite"].dropna())]
    niveaux = filtres[1].multiselect("Niveau d'aptitude", niveaux_presents, default=niveaux_presents)
    aptitude_min = filtres[2].slider("Aptitude minimale", 1.0, 5.0, 1.0, 0.1)
    surface_min = filtres[3].number_input("Surface minimale (ha)", 0.0, value=0.0, step=5.0)

    choix = zones[zones["strategie"].isin(strategies)
                  & (zones["priorite"].isna() | zones["priorite"].isin(niveaux))
                  & (zones["aptitude"].isna() | (zones["aptitude"] >= aptitude_min))
                  & (zones["surface_ha"] >= surface_min)]

    colonnes = st.columns(4)
    with colonnes[0]:
        kpi("Zones retenues", str(len(choix)), f"sur {len(zones)} au total")
    with colonnes[1]:
        kpi("Surface", f"{nombre(choix['surface_ha'].sum())} ha", "zones filtrées")
    with colonnes[2]:
        moyenne = choix["aptitude"].mean()
        kpi("Aptitude moyenne", f"{nombre(moyenne, 1)} / 5" if moyenne == moyenne else "—", "pondérée AHP")
    with colonnes[3]:
        forte = choix[choix["priorite"].isin(["très élevée", "élevée"])]
        kpi("Aptitude élevée ou très élevée", f"{nombre(forte['surface_ha'].sum())} ha", f"{len(forte)} zones")

    if choix.empty:
        st.info("Aucune zone ne correspond à ces filtres.")
        return

    st.write("")
    gauche, droite = st.columns(2, gap="large")
    with gauche:
        st.altair_chart(barres_surface(synthese(choix).query("zones > 0"), "Surface par intervention"),
                        width="stretch")
    with droite:
        st.altair_chart(nuage_aptitude(choix), width="stretch")
    st.altair_chart(nuage_pente_ndvi(choix), width="stretch")

    st.markdown("##### Liste des zones")
    tableau = choix.assign(intervention=[STRATEGIES[str(c)].nom for c in choix["strategie"]])
    colonnes_tableau = ["nom", "intervention", "aptitude", "priorite", "surface_ha", "pente", "ndvi",
                        "variation_ndvi", "altitude"]
    evenement = st.dataframe(
        tableau[colonnes_tableau], hide_index=True, width="stretch", height=380,
        on_select="rerun", selection_mode="single-row", key="tableau_zones",
        column_config={
            "nom": st.column_config.TextColumn("Zone", width="medium"),
            "intervention": st.column_config.TextColumn("Intervention"),
            "aptitude": st.column_config.ProgressColumn("Aptitude /5", min_value=0, max_value=5, format="%.1f"),
            "priorite": st.column_config.TextColumn("Niveau"),
            "surface_ha": st.column_config.NumberColumn("Surface (ha)", format="%.1f"),
            "pente": st.column_config.NumberColumn("Pente (%)", format="%.0f"),
            "ndvi": st.column_config.NumberColumn("NDVI 2025", format="%.2f"),
            "variation_ndvi": st.column_config.NumberColumn("Δ NDVI 2015-25", format="%+.2f"),
            "altitude": st.column_config.NumberColumn("Altitude (m)", format="%.0f"),
        },
    )
    lignes = evenement.selection.rows if evenement and evenement.selection else []
    if lignes:
        zone = tableau.iloc[lignes[0]]
        st.markdown(f"**{zone['nom']}** — {zone['justification'] or ''}")
        if st.button("📍 Localiser sur la carte", type="primary"):
            st.session_state["zone_selection"] = zone["map_id"]
            aller("carte")
    else:
        st.caption("Sélectionnez une ligne pour lire sa justification et la localiser sur la carte.")

    export = tableau[colonnes_tableau + ["justification"]]
    boutons = st.columns([1, 1, 3])
    boutons[0].download_button("⬇️ Exporter (CSV)", export.to_csv(index=False).encode("utf-8-sig"),
                               "zones_intervention.csv", "text/csv", width="stretch")
    boutons[1].download_button("⬇️ Exporter (GeoJSON)",
                               tableau[colonnes_tableau + ["justification", "geometry"]].to_json().encode("utf-8"),
                               "zones_intervention.geojson", "application/geo+json", width="stretch")
