"""Registre des pages pour les liens et redirections entre pages."""

from __future__ import annotations

from typing import Any

import streamlit as st

PAGES: dict[str, Any] = {}


def lien(cle: str, libelle: str, icone: str | None = None) -> None:
    if cle in PAGES:
        st.page_link(PAGES[cle], label=libelle, icon=icone, width="stretch")


def aller(cle: str) -> None:
    if cle in PAGES:
        st.switch_page(PAGES[cle])
