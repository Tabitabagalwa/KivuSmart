"""Climat futur : scénarios CMIP6 de la NASA (NEX-GDDP-CMIP6) et conséquences pour la restauration."""

from __future__ import annotations

from html import escape

import numpy as np
import pandas as pd
import streamlit as st

from KivuSmart.backend.app.analysis import climat as cl

from ..donnees import climat
from ..graphiques import SCENARIOS_COULEURS, cycle_saisonnier, trajectoires
from ..theme import entete, kpi, nombre

FENETRE_LISSAGE = 11  # années (moyenne glissante centrée, par modèle)


def _signe(valeur: float | None, decimales: int = 1, unite: str = "") -> str:
    if valeur is None or not np.isfinite(valeur):
        return "—"
    texte = nombre(abs(valeur), decimales)
    return ("+" if valeur > 0 else "−" if valeur < 0 else "") + texte + unite


def _intervalle(c, scenario, periode, indicateur, decimales=1, unite="") -> str:
    p10 = cl.valeur(c.synthese, scenario, periode, indicateur, "p10")
    p90 = cl.valeur(c.synthese, scenario, periode, indicateur, "p90")
    n = cl.valeur(c.synthese, scenario, periode, indicateur, "n_modeles")
    if p10 is None:
        return ""
    return f"{_signe(p10, decimales, unite)} à {_signe(p90, decimales, unite)} · {int(n or 0)} modèle(s)"


@st.cache_data(show_spinner=False)
def _series(annuel: pd.DataFrame, indicateur: str, relatif: bool) -> pd.DataFrame:
    """Anomalie annuelle lissée par modèle, puis médiane et centiles 10-90 % par scénario."""
    if indicateur not in annuel.columns:
        return pd.DataFrame(columns=["annee", "scenario", "mediane", "p10", "p90"])
    hist = annuel[annuel["scenario"] == "historical"]
    reference = hist[hist["annee"].between(*cl.REFERENCE)].groupby("modele")[indicateur].mean()
    lignes = []
    for scenario in [s for s in annuel["scenario"].unique() if s != "historical"]:
        for modele, groupe in annuel[annuel["scenario"].isin(["historical", scenario])].groupby("modele"):
            if modele not in reference.index:
                continue
            serie = groupe.set_index("annee")[indicateur].sort_index()
            serie = serie[~serie.index.duplicated(keep="last")]
            serie = serie.reindex(range(int(serie.index.min()), int(serie.index.max()) + 1))
            lisse = serie.rolling(FENETRE_LISSAGE, center=True, min_periods=5).mean()
            ref = reference[modele]
            anomalie = (lisse - ref) / ref * 100 if relatif else lisse - ref
            for annee, valeur in anomalie.dropna().items():
                lignes.append({"annee": annee, "scenario": "historical" if annee <= cl.REFERENCE[1] else scenario,
                               "modele": modele, "valeur": valeur})
    if not lignes:
        return pd.DataFrame(columns=["annee", "scenario", "mediane", "p10", "p90"])
    tableau = pd.DataFrame(lignes).drop_duplicates(["annee", "scenario", "modele"])
    resume = tableau.groupby(["annee", "scenario"])["valeur"].agg(
        mediane="median", p10=lambda v: np.percentile(v, 10), p90=lambda v: np.percentile(v, 90)).reset_index()
    return resume


def _carte_implication(message: dict[str, str]) -> None:
    st.markdown(
        f'<div class="kv-carte" style="--c:#2f6fdc"><div class="tete">{escape(message["theme"])}</div>'
        f'<div class="kv-note" style="margin-top:.15rem">Concerne : {escape(message["strategies"])}</div>'
        f'<div class="texte">{escape(message["texte"])}</div></div>', unsafe_allow_html=True)


def _absent() -> None:
    st.info("Les projections climatiques n'ont pas encore été téléchargées.")
    st.markdown("Depuis la racine du projet :")
    st.code("python scripts/telecharger_cmip6.py", language="powershell")
    st.caption("Source : NASA NEX-GDDP-CMIP6 (serveur THREDDS du NCCS, sans compte). Durée : environ une heure.")


def afficher() -> None:
    entete("Climat futur — scénarios CMIP6",
           "Projections de la NASA (NEX-GDDP-CMIP6) moyennées sur le territoire de Kalehe, et ce qu'elles "
           "changent pour le choix des interventions.")
    c = climat()
    if c is None:
        _absent()
        return

    scenarios = [s for s in ("ssp126", "ssp245", "ssp370", "ssp585") if s in set(c.synthese["scenario"])]
    if not scenarios:
        _absent()
        return
    g, d = st.columns([1.4, 1])
    with g:
        scenario = st.segmented_control(
            "Scénario d'émissions", scenarios, default="ssp245" if "ssp245" in scenarios else scenarios[0],
            format_func=lambda s: SCENARIOS_COULEURS[s][0], key="climat_scenario") or scenarios[0]
    with d:
        periode = st.segmented_control("Horizon", list(cl.PERIODES), default="2041-2060",
                                       key="climat_periode") or "2041-2060"
    st.caption(f"{c.meta.get('scenarios', {}).get(scenario, scenario)} · écart par rapport à "
               f"{c.meta.get('reference', '1995-2014')} · médiane des modèles (intervalle 10-90 %).")

    dt = cl.valeur(c.synthese, scenario, periode, "tas_c")
    colonnes = st.columns(5)
    tuiles = [
        ("Température moyenne", _signe(dt, 1, " °C"), _intervalle(c, scenario, periode, "tas_c", 1, " °C")),
        ("Pluie annuelle", _signe(cl.valeur(c.synthese, scenario, periode, "pluie_mm"), 0, " %"),
         _intervalle(c, scenario, periode, "pluie_mm", 0, " %")),
        ("Pluie journalière max.", _signe(cl.valeur(c.synthese, scenario, periode, "rx1day_mm"), 0, " %"),
         "Intensité des averses (érosion)"),
        ("Plus longue période sèche", _signe(cl.valeur(c.synthese, scenario, periode, "cdd_jours"), 1, " j"),
         "Jours consécutifs < 1 mm"),
        ("Remontée des étages", _signe(cl.decalage_altitudinal_m(dt) if dt is not None else None, 0, " m"),
         "Équivalent altitudinal du réchauffement (−6,5 °C/km)"),
    ]
    for colonne, (libelle, valeur, sous) in zip(colonnes, tuiles):
        with colonne:
            kpi(libelle, valeur, sous)

    st.write("")
    st.markdown("##### Ce que cela change pour la restauration")
    messages = cl.implications(c.synthese, scenario, periode)
    for debut in range(0, len(messages), 2):
        colonnes = st.columns(2)
        for colonne, message in zip(colonnes, messages[debut:debut + 2]):
            with colonne:
                _carte_implication(message)
        st.write("")

    st.markdown("##### Trajectoires 1995-2100")
    gauche, droite = st.columns(2, gap="large")
    with gauche:
        st.altair_chart(trajectoires(_series(c.annuel, "tas_c", False), "Écart de température",
                                     "Écart à 1995-2014 (°C)"), width="stretch")
    with droite:
        st.altair_chart(trajectoires(_series(c.annuel, "pluie_mm", True), "Écart de pluie annuelle",
                                     "Écart à 1995-2014 (%)", "+.0f"), width="stretch")
    st.caption(f"Moyenne glissante sur {FENETRE_LISSAGE} ans, par modèle ; ligne = médiane, bande = 10e-90e "
               "centile des modèles. Survolez une courbe pour lire les valeurs.")

    st.markdown("##### Saisons et indicateurs")
    gauche, droite = st.columns([1.1, 1], gap="large")
    with gauche:
        ref = c.mensuel[(c.mensuel["periode"] == "1995-2014") & (c.mensuel["scenario"] == "historical")]
        fut = c.mensuel[(c.mensuel["periode"] == periode) & (c.mensuel["scenario"] == scenario)]
        if not ref.empty and not fut.empty:
            nom_fut = f"{SCENARIOS_COULEURS[scenario][0]} {periode}"
            saison = pd.concat([
                ref.groupby("mois", as_index=False)["pluie_mm"].median().assign(serie="1995-2014",
                                                                                 couleur="#8a8985"),
                fut.groupby("mois", as_index=False)["pluie_mm"].median().assign(
                    serie=nom_fut, couleur=SCENARIOS_COULEURS[scenario][1]),
            ])
            st.altair_chart(cycle_saisonnier(saison, "Pluie mensuelle (médiane des modèles)", "pluie_mm",
                                             "Pluie (mm/mois)"), width="stretch")
    with droite:
        lignes = []
        for indicateur, (libelle, unite, mode) in cl.INDICATEURS.items():
            ligne = c.synthese[(c.synthese["scenario"] == scenario) & (c.synthese["periode"] == periode)
                               & (c.synthese["indicateur"] == indicateur)]
            if ligne.empty:
                continue
            l = ligne.iloc[0]
            u = " %" if mode == "relatif" else f" {unite}"
            d = 0 if mode == "relatif" or unite in ("mm", "jours/an", "jours") else 1
            lignes.append({"Indicateur": libelle,
                           "1995-2014": f"{nombre(l['reference'], 1)} {unite}",
                           "Écart médian": _signe(l["mediane"], d, u),
                           "Intervalle 10-90 %": f"{_signe(l['p10'], d, u)} à {_signe(l['p90'], d, u)}",
                           "Accord des modèles": f"{l['accord_signe'] * 100:.0f} %"})
        st.dataframe(pd.DataFrame(lignes), hide_index=True, width="stretch")
        st.caption("Accord des modèles : part des modèles qui projettent un changement de même signe que la médiane.")

    with st.expander("Contrôle avec les observations NASA POWER"):
        _validation(c)

    modeles = ", ".join(c.meta.get("modeles", []))
    st.markdown(
        f'<div class="kv-limites"><b>Source :</b> {escape(c.meta.get("source", ""))} — {escape(c.meta.get("resolution", ""))}. '
        f"Modèles : {escape(modeles)}. <b>Limites :</b> une cellule de 27 km mélange la rive du lac et les "
        "hautes terres ; les valeurs décrivent une tendance régionale. Les scénarios ne sont pas des prévisions : "
        "ils encadrent les futurs possibles selon les émissions mondiales. L'écart entre modèles (bande) fait "
        "partie du résultat.</div>", unsafe_allow_html=True)


def _validation(c) -> None:
    if c.power is None or c.power.empty:
        st.caption("Série NASA POWER absente (data/raw/power/power_mensuel_2006_2025.csv).")
        return
    annuel_power = c.power.groupby("annee").agg(pluie_mm=("pluie_mm", "sum"), tas_c=("T2M", "mean"),
                                                mois=("mois", "count"))
    annuel_power = annuel_power[annuel_power["mois"] == 12]
    commun = annuel_power[annuel_power.index <= cl.REFERENCE[1]]
    if commun.empty:
        st.caption("Pas d'année commune entre NASA POWER et la période historique des modèles.")
        return
    debut, fin = int(commun.index.min()), int(commun.index.max())
    hist = c.annuel[(c.annuel["scenario"] == "historical") & c.annuel["annee"].between(debut, fin)]
    modeles = hist.groupby("modele")[["pluie_mm", "tas_c"]].mean()
    st.markdown(
        f"Sur **{debut}-{fin}** (années communes) :\n"
        f"- **Pluie annuelle** : NASA POWER {nombre(commun['pluie_mm'].mean())} mm, modèles CMIP6 "
        f"{nombre(modeles['pluie_mm'].min())} à {nombre(modeles['pluie_mm'].max())} mm ;\n"
        f"- **Température moyenne** : NASA POWER {nombre(commun['tas_c'].mean(), 1)} °C, modèles "
        f"{nombre(modeles['tas_c'].min(), 1)} à {nombre(modeles['tas_c'].max(), 1)} °C.")
    st.caption("POWER est un point de grille (~50 km) et NEX-GDDP une moyenne sur Kalehe (cellules de 27 km) : "
               "un écart de quelques dixièmes de degré ou de 10-15 % de pluie est attendu. Les anomalies futures "
               "sont calculées par rapport à la période historique de chaque modèle, ce qui neutralise ce biais.")
