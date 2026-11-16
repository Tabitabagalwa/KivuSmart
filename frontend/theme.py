"""Identité visuelle de l'interface : couleurs, styles et petits composants HTML."""

from __future__ import annotations

from dataclasses import dataclass
from html import escape

import streamlit as st


@dataclass(frozen=True)
class Strategie:
    cle: str
    nom: str
    court: str
    couleur: str
    icone: str
    resume: str


# Palette catégorielle validée (vision normale et daltonisme, toutes paires : ΔE ≥ 8).
STRATEGIES: dict[str, Strategie] = {
    "rna": Strategie("rna", "Régénération naturelle assistée", "RNA", "#6250d6", "🌿",
                     "Laisser la forêt se reconstituer en lisière des aires protégées, à faible coût."),
    "agroforesterie": Strategie("agroforesterie", "Agroforesterie", "Agroforesterie", "#eda100", "🌾",
                                "Associer arbres et cultures sur les terroirs proches des villages."),
    "reboisement": Strategie("reboisement", "Reboisement pur", "Reboisement", "#14532d", "🌲",
                             "Planter les versants très pentus, hors usage agricole, contre l'érosion."),
    "prioritaire": Strategie("prioritaire", "Zones prioritaires", "Prioritaire", "#e34948", "⚠️",
                             "Champs sur versants raides ou défrichements récents : à traiter en premier."),
}
ORDRE = ("rna", "agroforesterie", "reboisement", "prioritaire")
BASELINE = Strategie("baseline", "Pratique actuelle", "Baseline", "#8a8985", "🏞️",
                     "Culture annuelle continue : référence de comparaison.")

TEXTE = "#111827"
TEXTE_2 = "#4b5563"
MUET = "#6b7280"
BORDURE = "#e5e7eb"
MARQUE = "#14532d"

# Palettes des rasters : séquentielles (une teinte, du clair au foncé) ou divergente
# (deux teintes autour d'un gris neutre) — jamais d'arc-en-ciel.
PALETTES_RASTER = {
    "altitude": ["#f7f1e8", "#dcc7a6", "#a97c50", "#5c3a1e"],
    "pente": ["#fff4ec", "#fdba8c", "#e8590c", "#7c2d12"],
    "vegetation": ["#f3faf0", "#a8d8a0", "#3c9a4e", "#14532d"],
    "eau": ["#eff6ff", "#9cc8fb", "#2f6fdc", "#1e3a8a"],
    "feux": ["#fef2f2", "#f9a8a8", "#d62f2f", "#7f1d1d"],
    "variation": ["#b42318", "#eba49b", "#e5e7eb", "#93cfa3", "#17803d"],
}

CSS = """
<style>
#MainMenu, footer, [data-testid="stDeployButton"] {visibility: hidden;}
.block-container {padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1480px;}
h1, h2, h3 {letter-spacing: -0.01em;}
[data-testid="stSidebar"] {background: #f4f7f5; border-right: 1px solid #e5e7eb;}
.kv-hero {background: linear-gradient(118deg, #0c3a20 0%, #14532d 48%, #1e6b58 100%);
  color: #fff; border-radius: 16px; padding: 26px 32px; margin-bottom: 1.1rem;}
.kv-hero h1 {color: #fff; font-size: 2rem; margin: 0; padding: 0;}
.kv-hero p {color: #d7f2e3; margin: .45rem 0 0; font-size: 1.02rem; max-width: 900px;}
.kv-hero .kv-tag {display: inline-block; background: rgba(255,255,255,.14); color: #fff;
  border: 1px solid rgba(255,255,255,.25); border-radius: 999px; padding: 2px 12px;
  font-size: .78rem; margin-bottom: .6rem;}
.kv-titre {margin: .2rem 0 .1rem; font-size: 1.55rem; font-weight: 700; color: #111827;}
.kv-sous-titre {color: #4b5563; margin-bottom: 1rem;}
.kv-kpi {background: #fff; border: 1px solid #e5e7eb; border-radius: 12px; padding: 14px 16px; height: 100%;}
.kv-kpi .lbl {font-size: .74rem; color: #6b7280; text-transform: uppercase; letter-spacing: .05em;}
.kv-kpi .val {font-size: 1.65rem; font-weight: 700; color: #111827; line-height: 1.25;}
.kv-kpi .sub {font-size: .8rem; color: #4b5563;}
.kv-carte {background: #fff; border: 1px solid #e5e7eb; border-top: 4px solid var(--c);
  border-radius: 12px; padding: 14px 16px; height: 100%;}
.kv-carte .tete {font-weight: 700; color: #111827; font-size: 1rem;}
.kv-carte .chiffre {font-size: 1.45rem; font-weight: 700; color: #111827; margin-top: .3rem;}
.kv-carte .texte {color: #4b5563; font-size: .86rem; margin-top: .3rem;}
.kv-puce {display: inline-flex; align-items: center; gap: 6px; padding: 2px 10px 2px 8px;
  border-radius: 999px; font-size: .78rem; font-weight: 600; color: #111827;
  background: color-mix(in srgb, var(--c) 14%, white); border: 1px solid color-mix(in srgb, var(--c) 45%, white);}
.kv-puce i {width: 9px; height: 9px; border-radius: 50%; background: var(--c); display: inline-block;}
.kv-etape {background: #fff; border: 1px solid #e5e7eb; border-radius: 12px; padding: 12px 14px; height: 100%;}
.kv-etape .num {display: inline-flex; width: 26px; height: 26px; border-radius: 50%; background: #14532d;
  color: #fff; align-items: center; justify-content: center; font-weight: 700; font-size: .85rem;}
.kv-etape .tete {font-weight: 700; color: #111827; margin-top: .4rem;}
.kv-etape .texte {color: #4b5563; font-size: .85rem;}
.kv-fiche {background: #fff; border: 1px solid #e5e7eb; border-left: 5px solid var(--c);
  border-radius: 12px; padding: 14px 16px;}
.kv-fiche table {width: 100%; font-size: .86rem; border-collapse: collapse; margin-top: .4rem;}
.kv-fiche td {padding: 4px 0; border-bottom: 1px solid #f1f2f4; color: #111827;}
.kv-fiche td:first-child {color: #6b7280; width: 52%;}
.kv-note {font-size: .8rem; color: #6b7280;}
.kv-limites {background: #f0f7ff; border: 1px solid #c9defa; border-radius: 12px; padding: 14px 18px; color: #1e3a5f;}
</style>
"""


def appliquer_theme() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def hero(titre: str, texte: str, etiquette: str | None = None) -> None:
    tag = f'<div class="kv-tag">{escape(etiquette)}</div>' if etiquette else ""
    st.markdown(f'<div class="kv-hero">{tag}<h1>{escape(titre)}</h1><p>{escape(texte)}</p></div>',
                unsafe_allow_html=True)


def entete(titre: str, sous_titre: str) -> None:
    st.markdown(f'<div class="kv-titre">{escape(titre)}</div>'
                f'<div class="kv-sous-titre">{escape(sous_titre)}</div>', unsafe_allow_html=True)


def kpi(libelle: str, valeur: str, sous_texte: str = "") -> None:
    st.markdown(f'<div class="kv-kpi"><div class="lbl">{escape(libelle)}</div>'
                f'<div class="val">{escape(valeur)}</div><div class="sub">{escape(sous_texte)}</div></div>',
                unsafe_allow_html=True)


def puce(strategie: Strategie, texte: str | None = None) -> str:
    return (f'<span class="kv-puce" style="--c:{strategie.couleur}"><i></i>'
            f'{escape(texte or strategie.court)}</span>')


def carte_strategie(strategie: Strategie, chiffre: str, texte: str) -> None:
    st.markdown(f'<div class="kv-carte" style="--c:{strategie.couleur}">'
                f'<div class="tete">{strategie.icone} {escape(strategie.nom)}</div>'
                f'<div class="chiffre">{escape(chiffre)}</div><div class="texte">{escape(texte)}</div></div>',
                unsafe_allow_html=True)


def etape(numero: int, titre: str, texte: str) -> None:
    st.markdown(f'<div class="kv-etape"><span class="num">{numero}</span>'
                f'<div class="tete">{escape(titre)}</div><div class="texte">{escape(texte)}</div></div>',
                unsafe_allow_html=True)


def nombre(valeur: float, decimales: int = 0) -> str:
    """Format français : espace fine comme séparateur de milliers, virgule décimale."""
    texte = f"{valeur:,.{decimales}f}".replace(",", " ").replace(".", ",")
    return texte


def theme_altair() -> dict:
    """Axes et grilles discrets pour les graphiques Altair."""
    return {
        "config": {
            "font": "Source Sans Pro, system-ui, sans-serif",
            "view": {"stroke": None},
            "axis": {"labelColor": TEXTE_2, "titleColor": TEXTE_2, "gridColor": "#eef0f2",
                     "domainColor": BORDURE, "tickColor": BORDURE, "labelFontSize": 12, "titleFontSize": 12},
            "legend": {"labelColor": TEXTE_2, "titleColor": TEXTE_2, "labelFontSize": 12, "orient": "top"},
            "title": {"color": TEXTE, "fontSize": 14, "anchor": "start"},
        }
    }
