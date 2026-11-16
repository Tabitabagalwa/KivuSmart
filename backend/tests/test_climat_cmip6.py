"""Tests des projections CMIP6 (NEX-GDDP), des indicateurs climatiques et des sources ouvertes."""

from __future__ import annotations

import io
import unittest

import httpx
import numpy as np
import pandas as pd
from scipy.io import netcdf_file
from shapely.geometry import box

from KivuSmart.backend.app.analysis import climat
from KivuSmart.backend.app.services.data_ingestion import sources_ouvertes as so
from KivuSmart.backend.app.services.geospatial.zonage import Grille
from KivuSmart.backend.app.services.nasa import cmip6
from KivuSmart.backend.app.services.nasa.client import NasaHttpClient
from KivuSmart.backend.app.services.nasa.errors import NasaApiError
from KivuSmart.backend.app.services.planetary_computer import stac


def netcdf3(variable: str, valeurs: np.ndarray, lat, lon) -> bytes:
    tampon = io.BytesIO()
    f = netcdf_file(tampon, "w")
    f.createDimension("time", valeurs.shape[0])
    f.createDimension("lat", len(lat))
    f.createDimension("lon", len(lon))
    f.createVariable("lat", "f8", ("lat",))[:] = lat
    f.createVariable("lon", "f8", ("lon",))[:] = lon
    v = f.createVariable(variable, "f4", ("time", "lat", "lon"))
    v[:] = valeurs
    v._FillValue = np.float32(1e20)
    f.flush()
    contenu = tampon.getvalue()
    f.close()
    return contenu


class Cmip6Tests(unittest.TestCase):
    def test_lecture_et_conversion_des_unites(self):
        valeurs = np.full((365, 2, 2), 1e-4, dtype="float32")
        valeurs[0, 0, 0] = 1e20
        grille = cmip6.lire(netcdf3("pr", valeurs, [-2.375, -2.125], [28.625, 388.875 - 360]), "pr", 2050)
        self.assertTrue(np.isnan(grille.valeurs[0, 0, 0]))
        self.assertAlmostEqual(float(grille.valeurs[1, 0, 0]), 8.64, places=3)  # kg m-2 s-1 -> mm/jour
        t = cmip6.lire(netcdf3("tas", np.full((360, 1, 1), 293.15, "float32"), [-2.1], [28.9]), "tas", 2050)
        self.assertAlmostEqual(float(t.valeurs.mean()), 20.0, places=3)

    def test_calendriers(self):
        self.assertEqual(np.bincount(cmip6.mois_des_jours(366))[2], 29)
        self.assertEqual(np.bincount(cmip6.mois_des_jours(365))[2], 28)
        self.assertEqual(set(np.bincount(cmip6.mois_des_jours(360))[1:]), {30})

    def test_poids_et_moyenne_de_zone(self):
        lat, lon = np.array([-2.375, -2.125]), np.array([28.625, 28.875])
        zone = box(28.5, -2.5, 28.75, -2.0)          # couvre entièrement la colonne ouest seulement
        poids = cmip6.poids_cellules(lat, lon, zone)
        np.testing.assert_allclose(poids, [[1, 0], [1, 0]])
        valeurs = np.array([[[1.0, 9.0], [np.nan, 9.0]]])
        self.assertEqual(float(cmip6.moyenne_zone(valeurs, poids)[0]), 1.0)  # NaN ignoré, poids renormalisés

    def test_chemins_et_emprise_valides(self):
        with self.assertRaises(ValueError):
            cmip6.dossier("../etc", "ssp245", "r1i1p1f1", "pr")
        with self.assertRaises(ValueError):
            cmip6.dossier("MRI-ESM2-0", "ssp999", "r1i1p1f1", "pr")
        with self.assertRaises(ValueError):
            cmip6.Emprise(30, -2, 29, -1).valider()
        client = NasaHttpClient("test", transport=httpx.MockTransport(lambda r: httpx.Response(200)))
        with self.assertRaises(ValueError):
            cmip6.telecharger(client, "../../secret.nc", "pr", cmip6.Emprise(28, -3, 29, -2))

    def test_catalogue_garde_la_derniere_version(self):
        xml = ('<catalog><dataset name="pr_day_M_ssp245_r1i1p1f1_gn_2050.nc" '
               'urlPath="AMES/NEX/GDDP-CMIP6/M/ssp245/r1i1p1f1/pr/pr_day_M_ssp245_r1i1p1f1_gn_2050.nc"/>'
               '<dataset name="pr_day_M_ssp245_r1i1p1f1_gn_2050_v2.0.nc" '
               'urlPath="AMES/NEX/GDDP-CMIP6/M/ssp245/r1i1p1f1/pr/pr_day_M_ssp245_r1i1p1f1_gn_2050_v2.0.nc"/>'
               '</catalog>')
        client = NasaHttpClient("test", transport=httpx.MockTransport(lambda r: httpx.Response(200, text=xml)))
        fichiers = cmip6.lister_fichiers(client, "M", "ssp245", "r1i1p1f1", "pr")
        self.assertTrue(fichiers[2050].endswith("_v2.0.nc"))

    def test_reponse_non_netcdf_refusee(self):
        client = NasaHttpClient("test", transport=httpx.MockTransport(lambda r: httpx.Response(200, text="<html>")))
        with self.assertRaises(NasaApiError):
            cmip6.telecharger(client, "AMES/NEX/GDDP-CMIP6/M/x.nc", "pr", cmip6.Emprise(28, -3, 29, -2))


class ClimatTests(unittest.TestCase):
    def test_indicateurs_annuels(self):
        mois = cmip6.mois_des_jours(365)
        pr = np.zeros(365)
        pr[100], pr[200:205] = 45.0, 2.0
        tas = np.full(365, 19.0)
        r = climat.indicateurs_annuels(pr, tas, tas + 8, mois)
        self.assertEqual(r["pluie_mm"], 55.0)
        self.assertEqual(r["rx1day_mm"], 45.0)
        self.assertEqual(r["jours_pluie_forte"], 1.0)
        self.assertEqual(r["cdd_jours"], 160.0)       # du 206e au 365e jour
        self.assertEqual(r["pluie_saison_seche_mm"], 2.0 * 5)
        self.assertEqual(r["jours_chauds"], 0.0)

    def _annuel(self):
        lignes = []
        for modele, rechauffement in (("A", 1.0), ("B", 2.0), ("C", 3.0)):
            for annee in range(1995, 2015):
                lignes.append({"modele": modele, "scenario": "historical", "annee": annee,
                               "tas_c": 19.0, "pluie_mm": 1500.0})
            for annee in range(2041, 2061):
                lignes.append({"modele": modele, "scenario": "ssp245", "annee": annee,
                               "tas_c": 19.0 + rechauffement, "pluie_mm": 1500.0 * (1 + rechauffement / 100)})
        return pd.DataFrame(lignes)

    def test_anomalies_et_synthese(self):
        table = climat.anomalies(self._annuel())
        synthese = climat.synthese_ensemble(table)
        self.assertAlmostEqual(climat.valeur(synthese, "ssp245", "2041-2060", "tas_c"), 2.0)
        self.assertAlmostEqual(climat.valeur(synthese, "ssp245", "2041-2060", "pluie_mm"), 2.0)  # en %
        self.assertEqual(climat.valeur(synthese, "ssp245", "2041-2060", "tas_c", "accord_signe"), 1.0)
        self.assertIsNone(climat.valeur(synthese, "ssp585", "2041-2060", "tas_c"))

    def test_implications_explicites(self):
        synthese = climat.synthese_ensemble(climat.anomalies(self._annuel()))
        messages = climat.implications(synthese, "ssp245", "2041-2060")
        essences = next(m for m in messages if m["theme"] == "Choix des essences")
        self.assertIn("308 m", essences["texte"])      # 2,0 °C / 6,5 °C par km
        self.assertAlmostEqual(climat.decalage_altitudinal_m(1.3), 200.0)


class SourcesTests(unittest.TestCase):
    def test_hotes_autorises(self):
        with self.assertRaises(so.SourceOuverteErreur):
            so._verifier("https://exemple.com/x")
        with self.assertRaises(so.SourceOuverteErreur):
            so._verifier("http://overpass-api.de/api/interpreter")
        so._verifier("https://overpass-api.de/api/interpreter")
        with self.assertRaises(stac.PlanetaryComputerErreur):
            stac._verifier("https://evil.blob.core.windows.net/x")          # stockage refusé hors COG signé
        stac._verifier("https://landsateuwest.blob.core.windows.net/x.tif", stockage=True)

    def test_grille_utm(self):
        g = Grille((28.70, -2.42, 28.92, -2.22), 30.0)
        self.assertEqual(g.forme, (739, 818))
        self.assertEqual(g.distance(np.zeros(g.forme, bool)).max(), 1e6)   # aucun objet : distance « infinie »


from KivuSmart.backend.tests.support import ApiTestCase  # noqa: E402


class Cmip6ApiTests(ApiTestCase):
    def test_synthese_absente_puis_filtree(self):
        reponse = self.client.get("/api/v1/nasa/cmip6/synthese")
        self.assertEqual(reponse.status_code, 404)

        dossier = self.data_dir / "processed" / "climat"
        dossier.mkdir(parents=True, exist_ok=True)
        synthese = climat.synthese_ensemble(climat.anomalies(ClimatTests()._annuel()))
        synthese.to_csv(dossier / "cmip6_synthese.csv", index=False)
        (dossier / "cmip6_meta.json").write_text('{"source": "NASA NEX-GDDP-CMIP6", "grille": {}}', encoding="utf-8")

        reponse = self.client.get("/api/v1/nasa/cmip6/synthese", params={"scenario": "ssp245", "periode": "2041-2060"})
        self.assertEqual(reponse.status_code, 200, reponse.text)
        corps = reponse.json()
        self.assertNotIn("grille", corps["meta"])
        self.assertEqual({l["scenario"] for l in corps["synthese"]}, {"ssp245"})
        self.assertTrue(any(m["theme"] == "Choix des essences" for m in corps["implications"]))
        self.assertEqual(self.client.get("/api/v1/nasa/cmip6/synthese", params={"scenario": "x"}).status_code, 422)


if __name__ == "__main__":
    unittest.main()
