from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from KivuSmart.backend.app.core.config import PROJECT_ROOT
from KivuSmart.backend.app.decision_engine import ahp_weights, evaluate_parcel
from KivuSmart.backend.app.decision_engine.criteria import decreasing, increasing, trapezoid
from KivuSmart.backend.app.decision_engine.rules import DEFAULT_RULES, load_rules
from KivuSmart.backend.app.decision_engine.scoring_engine import STRATEGY_MODULES
from KivuSmart.backend.engine import build_sample_parcels, recommend_restoration_strategy

REFERENCE = PROJECT_ROOT / "data" / "reference" / "referentiel_kivu.json"


class AhpTests(unittest.TestCase):
    def test_known_consistent_matrix(self):
        result = ahp_weights([[1, 3, 5], [1 / 3, 1, 3], [1 / 5, 1 / 3, 1]])
        self.assertAlmostEqual(sum(result.weights), 1.0, places=5)
        self.assertEqual([round(w, 3) for w in result.weights], [0.637, 0.258, 0.105])
        self.assertLess(result.consistency_ratio, 0.05)
        self.assertTrue(result.consistent)

    def test_inconsistent_matrix_is_flagged(self):
        result = ahp_weights([[1, 9, 1 / 9], [1 / 9, 1, 9], [9, 1 / 9, 1]])
        self.assertFalse(result.consistent)

    def test_invalid_matrices_are_rejected(self):
        for matrix in ([[1, 2], [2, 1]], [[1, 0], [0, 1]], [[1, 20], [1 / 20, 1]], [[1, 2, 3]], [[2]]):
            with self.subTest(matrix=matrix), self.assertRaises(ValueError):
                ahp_weights(matrix)

    def test_default_strategy_matrices_are_consistent(self):
        for module in STRATEGY_MODULES.values():
            with self.subTest(strategy=module.STRATEGY_ID):
                self.assertTrue(ahp_weights(module.DEFAULT_PAIRWISE).consistent)


class MembershipTests(unittest.TestCase):
    def test_functions(self):
        self.assertEqual(increasing(5, 0, 10), 0.5)
        self.assertEqual(decreasing(800, 800, 2000), 1.0)
        self.assertEqual(decreasing(2000, 800, 2000), 0.0)
        self.assertEqual(trapezoid(25, 10, 15, 35, 40), 1.0)
        self.assertAlmostEqual(trapezoid(12.5, 10, 15, 35, 40), 0.5)
        self.assertIsNone(increasing(None, 0, 1))


class EvaluationTests(unittest.TestCase):
    def test_matches_legacy_rule_engine_on_sample_parcels(self):
        for parcel in build_sample_parcels():
            with self.subTest(parcel=parcel["id"]):
                result = evaluate_parcel(parcel, reference_path=REFERENCE)
                legacy = recommend_restoration_strategy(parcel)["strategy_id"]
                self.assertEqual(result["decision_status"], "recommended")
                self.assertEqual(result["recommended"]["strategy_id"], legacy)
                self.assertTrue(result["legacy_rule_engine"]["agrees"])
                self.assertTrue(result["recommended"]["species"])

    def test_missing_data_is_never_defaulted(self):
        result = evaluate_parcel({"slope_pct": 20})
        self.assertEqual(result["decision_status"], "insufficient_data")
        self.assertIsNone(result["recommended"])
        self.assertIsNone(result["legacy_rule_engine"])  # legacy engine would default to 0
        self.assertIn("Variable manquante : ndvi", result["information_gaps"])

    def test_low_suitability_keeps_baseline_as_reference(self):
        metrics = {"slope_pct": 28, "distance_to_forest_m": 5000, "ndvi": 0.7, "canopy_height_m": 0,
                   "cropland": False, "degraded": False, "water_buffer_m": 2000,
                   "distance_to_village_km": 10, "soil_moisture": 0.5}
        result = evaluate_parcel(metrics)
        self.assertEqual(result["decision_status"], "low_suitability")
        self.assertEqual(result["confidence"], 0.0)

    def test_resolution_gaps_are_reported(self):
        result = evaluate_parcel({**build_sample_parcels()[2]})
        self.assertTrue(any("SMAP" in gap for gap in result["information_gaps"]))

    def test_baseline_erosion_risk(self):
        high = evaluate_parcel(build_sample_parcels()[1])["baseline"]
        self.assertEqual(high["erosion_risk_level"], "élevé")


class RulesTests(unittest.TestCase):
    def test_referentiel_and_config_overrides(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp)
            (config / "restoration_rules.json").write_text(
                json.dumps({"rna": {"forest_distance_max_m": 500}}), encoding="utf-8")
            rules = load_rules(config, REFERENCE)
            self.assertEqual(rules["rna"]["forest_distance_max_m"], 500)
            self.assertEqual(rules["reboisement"]["slope_min_pct"], 35)
            self.assertEqual(DEFAULT_RULES["rna"]["forest_distance_max_m"], 800)  # defaults untouched

            (config / "restoration_rules.json").write_text(json.dumps({"rna": {"typo": 1}}), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_rules(config, REFERENCE)
            (config / "restoration_rules.json").write_text(json.dumps({"rna": {"slope_min_pct": "x"}}),
                                                           encoding="utf-8")
            with self.assertRaises(ValueError):
                load_rules(config, REFERENCE)

    def test_project_config_loads(self):
        rules = load_rules(PROJECT_ROOT / "config", REFERENCE)
        self.assertEqual(rules["agroforesterie"]["village_distance_max_km"], 2.0)


if __name__ == "__main__":
    unittest.main()
