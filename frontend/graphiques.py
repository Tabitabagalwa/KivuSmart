"""Graphiques Altair partagés (couleurs fixes par intervention, axes discrets)."""

from __future__ import annotations

import altair as alt
import pandas as pd

from .theme import ORDRE, STRATEGIES, TEXTE_2, nombre, theme_altair

NOMS = [STRATEGIES[c].nom for c in ORDRE]
COULEURS = [STRATEGIES[c].couleur for c in ORDRE]
ECHELLE = alt.Scale(domain=NOMS, range=COULEURS)


def _configurer(graphique: alt.Chart) -> alt.Chart:
    config = theme_altair()["config"]
    return (graphique.configure(font=config["font"]).configure_view(stroke=None)
            .configure_axis(**config["axis"]).configure_legend(**config["legend"])
            .configure_title(**config["title"]))


def barres_surface(synthese: pd.DataFrame, titre: str = "Surface identifiée par intervention") -> alt.Chart:
    """Barres horizontales, une par intervention, étiquetées directement."""
    donnees = synthese.copy()
    donnees["nom"] = [STRATEGIES[c].nom for c in donnees["strategie"]]
    donnees["etiquette"] = [f"{nombre(s)} ha · {n} zone{'s' if n > 1 else ''}"
                            for s, n in zip(donnees["surface_ha"], donnees["zones"])]
    base = alt.Chart(donnees).encode(
        y=alt.Y("nom:N", sort=NOMS, title=None, axis=alt.Axis(labelLimit=240, ticks=False, domain=False)),
        x=alt.X("surface_ha:Q", title="Surface (ha)", axis=alt.Axis(grid=True, tickCount=5)),
    )
    barres = base.mark_bar(cornerRadiusEnd=4, height=24).encode(
        color=alt.Color("nom:N", scale=ECHELLE, legend=None),
        tooltip=[alt.Tooltip("nom:N", title="Intervention"),
                 alt.Tooltip("surface_ha:Q", title="Surface (ha)", format=",.0f"),
                 alt.Tooltip("zones:Q", title="Zones"),
                 alt.Tooltip("aptitude:Q", title="Aptitude moyenne /5", format=".1f")],
    )
    etiquettes = base.mark_text(align="left", dx=6, color=TEXTE_2, fontSize=12).encode(text="etiquette:N")
    hauteur = max(140, 46 * len(donnees))
    return _configurer((barres + etiquettes).properties(title=titre, height=hauteur))


def nuage_aptitude(zones: pd.DataFrame) -> alt.Chart:
    """Distribution des aptitudes : un point par zone, taille = surface."""
    donnees = zones.assign(nom_strategie=[STRATEGIES[str(c)].nom for c in zones["strategie"]])
    donnees = donnees[["nom", "nom_strategie", "aptitude", "surface_ha", "priorite"]].dropna(subset=["aptitude"])
    return _configurer(alt.Chart(donnees).mark_circle(opacity=0.85, stroke="#ffffff", strokeWidth=1.5).encode(
        x=alt.X("aptitude:Q", title="Aptitude (1 = très faible, 5 = très élevée)", scale=alt.Scale(domain=[1, 5])),
        y=alt.Y("nom_strategie:N", sort=NOMS, title=None, axis=alt.Axis(labelLimit=240, ticks=False, domain=False)),
        size=alt.Size("surface_ha:Q", title="Surface (ha)", scale=alt.Scale(range=[60, 900])),
        color=alt.Color("nom_strategie:N", scale=ECHELLE, legend=None),
        tooltip=[alt.Tooltip("nom:N", title="Zone"), alt.Tooltip("aptitude:Q", title="Aptitude /5", format=".2f"),
                 alt.Tooltip("surface_ha:Q", title="Surface (ha)", format=",.1f"),
                 alt.Tooltip("priorite:N", title="Priorité")],
    ).properties(title="Aptitude des zones", height=max(160, 52 * donnees["nom_strategie"].nunique())))


def nuage_pente_ndvi(zones: pd.DataFrame) -> alt.Chart:
    """Pente moyenne contre NDVI 2025 : situe chaque zone dans son contexte physique."""
    donnees = zones.assign(nom_strategie=[STRATEGIES[str(c)].nom for c in zones["strategie"]])
    donnees = donnees[["nom", "nom_strategie", "pente", "ndvi", "surface_ha"]].dropna(subset=["pente", "ndvi"])
    presentes = [n for n in NOMS if n in set(donnees["nom_strategie"])]
    return _configurer(alt.Chart(donnees).mark_circle(opacity=0.85, stroke="#ffffff", strokeWidth=1.5).encode(
        x=alt.X("pente:Q", title="Pente moyenne (%)"),
        y=alt.Y("ndvi:Q", title="NDVI 2025 (végétation)"),
        size=alt.Size("surface_ha:Q", legend=None, scale=alt.Scale(range=[60, 700])),
        color=alt.Color("nom_strategie:N", title=None,
                        scale=alt.Scale(domain=presentes, range=[STRATEGIES[c].couleur for c in ORDRE
                                                                 if STRATEGIES[c].nom in presentes])),
        tooltip=[alt.Tooltip("nom:N", title="Zone"), alt.Tooltip("pente:Q", title="Pente (%)", format=".0f"),
                 alt.Tooltip("ndvi:Q", title="NDVI", format=".2f"),
                 alt.Tooltip("surface_ha:Q", title="Surface (ha)", format=",.1f")],
    ).properties(title="Pente et végétation des zones", height=330))


# ------------------------------------------------------------------------- climat
# Palette des scénarios validée (OKLab, simulations deutéranopie / protanopie /
# tritanopie : ΔE ≥ 11 pour toutes les paires) ; l'ambre, peu contrasté sur blanc,
# est toujours accompagné d'étiquettes directes et d'un tableau.
SCENARIOS_COULEURS = {
    "historical": ("Historique", "#8a8985"),
    "ssp126": ("SSP1-2.6", "#2f6fdc"),
    "ssp245": ("SSP2-4.5", "#eda100"),
    "ssp370": ("SSP3-7.0", "#7c3aed"),
    "ssp585": ("SSP5-8.5", "#b42318"),
}


def echelle_scenarios(presents: list[str]) -> alt.Scale:
    cles = [c for c in SCENARIOS_COULEURS if c in presents]
    return alt.Scale(domain=[SCENARIOS_COULEURS[c][0] for c in cles], range=[SCENARIOS_COULEURS[c][1] for c in cles])


def trajectoires(donnees: pd.DataFrame, titre: str, titre_axe: str, format_valeur: str = "+.1f") -> alt.Chart:
    """Médiane multi-modèles (ligne) et intervalle 10-90 % (bande) par scénario.

    ``donnees`` : colonnes ``annee, scenario, mediane, p10, p90``.
    """
    d = donnees.copy()
    d["nom"] = [SCENARIOS_COULEURS[s][0] for s in d["scenario"]]
    echelle = echelle_scenarios(list(d["scenario"].unique()))
    base = alt.Chart(d).encode(x=alt.X("annee:Q", title=None, axis=alt.Axis(format="d", tickCount=9, grid=False)))
    bande = base.mark_area(opacity=0.16).encode(y="p10:Q", y2="p90:Q", color=alt.Color("nom:N", scale=echelle,
                                                                                          legend=None))
    ligne = base.mark_line(strokeWidth=2).encode(
        y=alt.Y("mediane:Q", title=titre_axe, axis=alt.Axis(grid=True, tickCount=6)),
        color=alt.Color("nom:N", scale=echelle, title=None, legend=alt.Legend(orient="top")))
    survol = alt.selection_point(nearest=True, on="pointerover", fields=["annee"], empty=False)
    points = base.mark_circle(size=70, stroke="#ffffff", strokeWidth=2).encode(
        y="mediane:Q", color=alt.Color("nom:N", scale=echelle, legend=None),
        opacity=alt.condition(survol, alt.value(1), alt.value(0)),
        tooltip=[alt.Tooltip("annee:Q", title="Année", format="d"), alt.Tooltip("nom:N", title="Scénario"),
                 alt.Tooltip("mediane:Q", title="Médiane", format=format_valeur),
                 alt.Tooltip("p10:Q", title="10e centile", format=format_valeur),
                 alt.Tooltip("p90:Q", title="90e centile", format=format_valeur)],
    ).add_params(survol)
    repere = base.mark_rule(color="#9ca3af", strokeWidth=1).encode(
        opacity=alt.condition(survol, alt.value(0.8), alt.value(0))).transform_filter(survol)
    fin = d.sort_values("annee").groupby("scenario").tail(1)
    etiquettes = alt.Chart(fin).mark_text(align="left", dx=6, fontSize=11, color=TEXTE_2).encode(
        x="annee:Q", y="mediane:Q", text="nom:N")
    zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="#d1d5db", strokeDash=[4, 3]).encode(y="y:Q")
    return _configurer((zero + bande + ligne + repere + points + etiquettes).properties(title=titre, height=320))


def cycle_saisonnier(donnees: pd.DataFrame, titre: str, champ: str, titre_axe: str) -> alt.Chart:
    """Climatologie mensuelle : référence 1995-2014 contre période future (deux séries)."""
    d = donnees.copy()
    mois = ["janv.", "févr.", "mars", "avr.", "mai", "juin", "juil.", "août", "sept.", "oct.", "nov.", "déc."]
    d["mois_nom"] = [mois[m - 1] for m in d["mois"]]
    domaine = list(dict.fromkeys(d["serie"]))
    couleurs = [d.loc[d["serie"] == s, "couleur"].iloc[0] for s in domaine]
    base = alt.Chart(d).encode(
        x=alt.X("mois_nom:N", sort=mois, title=None, axis=alt.Axis(labelAngle=0, ticks=False)),
        color=alt.Color("serie:N", scale=alt.Scale(domain=domaine, range=couleurs), title=None,
                        legend=alt.Legend(orient="top")))
    ligne = base.mark_line(strokeWidth=2).encode(y=alt.Y(f"{champ}:Q", title=titre_axe))
    points = base.mark_circle(size=80, stroke="#ffffff", strokeWidth=2, opacity=1).encode(
        y=f"{champ}:Q",
        tooltip=[alt.Tooltip("mois_nom:N", title="Mois"), alt.Tooltip("serie:N", title="Période"),
                 alt.Tooltip(f"{champ}:Q", title=titre_axe, format=".1f")])
    return _configurer((ligne + points).properties(title=titre, height=280))


def barres_facteurs(facteurs: pd.DataFrame, couleur: str, titre: str, champ: str = "valeur",
                    titre_axe: str = "Poids (%)", domaine: list[float] | None = None) -> alt.Chart:
    """Barres horizontales d'une seule série (la couleur de l'intervention, pas de légende)."""
    base = alt.Chart(facteurs).encode(
        y=alt.Y("facteur:N", sort=None, title=None, axis=alt.Axis(labelLimit=260, ticks=False, domain=False)),
        x=alt.X(f"{champ}:Q", title=titre_axe, scale=alt.Scale(domain=domaine) if domaine else alt.Undefined),
    )
    barres = base.mark_bar(cornerRadiusEnd=4, height=18, color=couleur).encode(
        tooltip=[alt.Tooltip("facteur:N", title="Facteur"), alt.Tooltip(f"{champ}:Q", title=titre_axe, format=".1f")])
    etiquettes = base.mark_text(align="left", dx=5, color=TEXTE_2, fontSize=11).encode(
        text=alt.Text(f"{champ}:Q", format=".1f"))
    return _configurer((barres + etiquettes).properties(title=titre, height=34 * len(facteurs)))
