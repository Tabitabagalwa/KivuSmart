"""Tests de la nouvelle interface : règles d'aptitude, carte, données et pages."""

from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from KivuSmart.backend.app.decision_engine.aptitude import aptitude, classer, criteres, evaluer

RACINE = Path(__file__).resolve().parents[2]
DEMO_DISPONIBLE = (RACINE / "config" / "simulation" / "map_layers.json").is_file()


class AptitudeTests(unittest.TestCase):
    base = dict(pente=30, occupation=2, dans_pnkb=False, d_pnkb=800, d_champs=1200, d_village=2600,
                d_route=2000, d_riviere=700, ndvi=0.58, variation_ndvi=0.0, couvert=45, feux=1)

    def test_classer_suit_les_bornes(self):
        self.assertEqual(classer(np.array([0, 500, 2000, 9000]), (500, 1500, 3000), (5, 4, 3, 2)).tolist(),
                         [5, 4, 3, 2])

    def test_regles_du_document(self):
        self.assertEqual(evaluer(self.base)["recommande"], "rna")
        self.assertEqual(evaluer({**self.base, "pente": 45, "occupation": 3, "d_pnkb": 6000})["recommande"],
                         "reboisement")
        agro = {**self.base, "pente": 12, "occupation": 4, "d_champs": 0, "d_village": 600, "d_pnkb": 7000}
        self.assertEqual(evaluer(agro)["recommande"], "agroforesterie")
        self.assertIsNone(evaluer({**agro, "occupation": 5})["recommande"])  # eau exclue
        self.assertIsNone(evaluer({**self.base, "pente": 45, "occupation": 3, "dans_pnkb": True, "d_pnkb": 0})
                          ["recommande"])  # reboisement pur interdit dans le PNKB

    def test_champ_sur_versant_raide_est_prioritaire(self):
        resultat = evaluer({**self.base, "pente": 42, "occupation": 4, "d_champs": 0})
        self.assertTrue(resultat["prioritaire"])
        self.assertIsNone(resultat["recommande"])

    def test_tableaux_et_scalaires_coherents(self):
        tableaux = {k: np.array([v, v]) for k, v in self.base.items()}
        self.assertTrue(criteres(tableaux)["rna"].all())
        self.assertAlmostEqual(float(aptitude("rna", self.base)), float(aptitude("rna", tableaux)[0]))


@unittest.skipUnless(DEMO_DISPONIBLE, "jeu de démonstration non généré")
class InterfaceTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {"KIVU_DONNEES": "demo"})
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_donnees_et_carte(self):
        from KivuSmart.frontend.carte import coloriser, construire_carte
        from KivuSmart.frontend.donnees import charger

        donnees = charger()
        self.assertTrue(donnees.demo)
        self.assertEqual(set(donnees.zones["strategie"].astype(str)),
                         {"rna", "agroforesterie", "reboisement", "prioritaire"})
        self.assertGreater(donnees.surface_zone_ha, 0)
        raster = next(r for r in donnees.rasters if not r.get("categorical"))
        html = construire_carte(donnees, donnees.zones, rasters=[raster],
                                selection=donnees.zones["map_id"].iloc[0]).get_root().render()
        for attendu in ("Imagerie satellite", "Reboisement pur", "kv-leg", "fullscreen"):
            self.assertIn(attendu, html)

        image, legende = coloriser({"palette": "variation"}, np.array([[-0.2, 0.0, 0.2, np.nan]]))
        self.assertEqual(legende["min"], -legende["max"])  # palette divergente centrée sur zéro
        self.assertEqual(image[0, 3, 3], 0)  # pas de donnée = transparent

    def test_pages_sans_erreur(self):
        from streamlit.testing.v1 import AppTest

        for vue in ("accueil", "carte_page", "analyse", "strategies", "simulateur", "methodologie"):
            with self.subTest(vue=vue):
                script = (
                    "import streamlit as st\n"
                    "from frontend.theme import appliquer_theme\n"
                    f"from frontend.vues import {vue}\n"
                    "appliquer_theme()\n"
                    f"{vue}.afficher()\n"
                )
                at = AppTest.from_string(script, default_timeout=300).run()
                self.assertEqual([e.value for e in at.exception], [])


if __name__ == "__main__":
    unittest.main()
