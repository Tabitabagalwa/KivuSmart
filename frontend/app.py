from __future__ import annotations

from KivuSmart.frontend.vues import accueil, analyse, carte_page, climat, methodologie, simulateur

try:
    import streamlit as st
except ImportError as exc:  # pragma: no cover - Streamlit is installed for the UI
    raise RuntimeError(
        "Streamlit is required to run the Kivu Smart interface. "
        "Install dependencies from requirements.txt."
    ) from exc

from KivuSmart.frontend import navigation
from KivuSmart.frontend.theme import appliquer_theme
from KivuSmart.frontend.vues import strategies


def main() -> None:
    st.set_page_config(page_title="Kivu Smart — Restauration des paysages", page_icon="🌳",
                       layout="wide", initial_sidebar_state="expanded")
    appliquer_theme()

    pages = {
        "accueil": st.Page(accueil.afficher, title="Tableau de bord", icon="🏠", url_path="accueil", default=True),
        "carte": st.Page(carte_page.afficher, title="Carte des zones", icon="🗺️", url_path="carte"),
        "analyse": st.Page(analyse.afficher, title="Analyse des zones", icon="📊", url_path="analyse"),
        "strategies": st.Page(strategies.afficher, title="Comparer les stratégies", icon="⚖️", url_path="strategies"),
        "simulateur": st.Page(simulateur.afficher, title="Simulateur de décision", icon="🧪", url_path="simulateur"),
        "climat": st.Page(climat.afficher, title="Climat futur (CMIP6)", icon="🌡️", url_path="climat"),
        "methodologie": st.Page(methodologie.afficher, title="Méthodologie et données", icon="📘",
                                url_path="methodologie"),
    }
    navigation.PAGES.clear()
    navigation.PAGES.update(pages)

    with st.sidebar:
        st.markdown(
            '<div class="kv-sidebar-brand">'
            '<div class="kv-sidebar-title">🌳 Kivu Smart</div>'
            '<div class="kv-sidebar-subtitle">Restauration intelligente des paysages · Sud-Kivu</div>'
            "</div>",
            unsafe_allow_html=True,
        )

    menu = st.navigation({
        "Présentation": [pages["accueil"], pages["strategies"]],
        "Exploration": [pages["carte"], pages["analyse"], pages["simulateur"], pages["climat"]],
        "Référence": [pages["methodologie"]],
    })
    menu.run()

    with st.sidebar:
        st.markdown("---")
        st.caption("NASA Space Apps Challenge 2026 · Bukavu\n\nDéfi « Field Shift: Adapting Farms with NASA Data »")


if __name__ == "__main__":
    main()
