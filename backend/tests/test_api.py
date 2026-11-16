"""End-to-end tests of the FastAPI backend (SQLite, fake NASA services, offline)."""

from __future__ import annotations

import json
import unittest

import httpx
import numpy as np
from pydantic import ValidationError

from KivuSmart.backend.app.core.config import Settings
from KivuSmart.backend.app.db.models import NasaRequestLog, Observation
from KivuSmart.backend.tests.support import AUTH, ApiTestCase, demo_grid, power_payload, write_raster


class HealthAndSecurityTests(ApiTestCase):
    def test_health_and_readiness(self):
        self.assertEqual(self.client.get("/health").json()["status"], "ok")
        ready = self.client.get("/health/ready").json()
        self.assertEqual(ready["status"], "ready")
        self.assertEqual(ready["database"]["dialect"], "sqlite")

    def test_security_headers_are_set(self):
        response = self.client.get("/api/v1/parcels")
        self.assertEqual(response.headers["X-Content-Type-Options"], "nosniff")
        self.assertEqual(response.headers["X-Frame-Options"], "DENY")
        self.assertIn("default-src 'none'", response.headers["Content-Security-Policy"])

    def test_writes_require_a_valid_api_key(self):
        self.assertEqual(self.client.post("/api/v1/imports/demo").status_code, 401)
        self.assertEqual(
            self.client.post("/api/v1/imports/demo", headers={"X-API-Key": "wrong"}).status_code, 401
        )
        self.assertEqual(self.client.post("/api/v1/imports/demo", headers=AUTH).status_code, 200)

    def test_unexpected_errors_do_not_leak_internals(self):
        @self.app.get("/boom")
        def boom():
            raise RuntimeError("secret connection string")

        from fastapi.testclient import TestClient

        with TestClient(self.app, raise_server_exceptions=False) as client:
            response = client.get("/boom")
        self.assertEqual(response.status_code, 500)
        self.assertNotIn("secret", response.text)

    def test_invalid_path_parameter_is_rejected(self):
        self.assertEqual(self.client.get("/api/v1/parcels/..%2Fetc").status_code, 404)
        self.assertEqual(self.client.get("/api/v1/parcels/bad code!").status_code, 422)


class RateLimitTests(ApiTestCase):
    settings_overrides = {"rate_limit_per_minute": 3}

    def test_rate_limit_returns_429_but_health_is_exempt(self):
        codes = [self.client.get("/api/v1/parcels").status_code for _ in range(4)]
        self.assertEqual(codes, [200, 200, 200, 429])
        self.assertEqual(self.client.get("/health").status_code, 200)


class UploadLimitTests(ApiTestCase):
    settings_overrides = {"max_upload_mb": 0.01}

    def test_oversized_upload_is_rejected(self):
        payload = b"x" * 200_000
        response = self.client.post(
            "/api/v1/imports/parcels", headers=AUTH, data={"source": "Relevé terrain"},
            files={"file": ("big.geojson", payload, "application/geo+json")},
        )
        self.assertEqual(response.status_code, 413)


class ProductionSettingsTests(unittest.TestCase):
    def test_production_requires_key_postgres_and_explicit_cors(self):
        with self.assertRaises(ValidationError):
            Settings(_env_file=None, environment="production", api_key=None)
        with self.assertRaises(ValidationError):
            Settings(_env_file=None, environment="production", api_key="x" * 32,
                     database_url="sqlite:///x.db")
        with self.assertRaises(ValidationError):
            Settings(_env_file=None, environment="production", api_key="x" * 32, cors_origins=["*"])
        ok = Settings(_env_file=None, environment="production", api_key="x" * 32)
        self.assertEqual(ok.environment, "production")


class ParcelAndImportTests(ApiTestCase):
    def test_demo_import_is_flagged_and_idempotent(self):
        first = self.load_demo()
        self.assertEqual(first["parcels"]["created"], 3)
        second = self.load_demo()
        self.assertEqual(second["parcels"]["updated"], 3)
        with self.app.state.database.session() as session:
            self.assertEqual(session.query(Observation).count(), first["metrics"]["created"])
        parcel = self.client.get("/api/v1/parcels/parcelle_01").json()
        self.assertTrue(parcel["is_demo"])
        self.assertAlmostEqual(parcel["area_ha"], 88.6, delta=1.0)

    def test_list_pagination_bbox_and_geojson(self):
        self.load_demo()
        page = self.client.get("/api/v1/parcels", params={"limit": 2}).json()
        self.assertEqual((page["total"], len(page["items"])), (3, 2))
        west = self.client.get("/api/v1/parcels", params={"bbox": "28.75,-2.335,28.765,-2.32"}).json()
        self.assertEqual([p["code"] for p in west["items"]], ["parcelle_03"])
        self.assertEqual(self.client.get("/api/v1/parcels", params={"bbox": "1,2,3"}).status_code, 422)
        geojson = self.client.get("/api/v1/parcels/geojson").json()
        self.assertEqual(geojson["type"], "FeatureCollection")
        self.assertEqual(geojson["features"][0]["geometry"]["type"], "MultiPolygon")
        self.assertEqual(self.client.get("/api/v1/parcels/unknown").status_code, 404)

    def test_parcel_upload_sanitizes_filename_and_rejects_bad_extensions(self):
        collection = {
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "properties": {"code": "P-1", "nom": "Colline A"},
                "geometry": {"type": "Polygon", "coordinates": [[
                    [28.80, -2.30], [28.81, -2.30], [28.81, -2.29], [28.80, -2.29], [28.80, -2.30]]]},
            }],
        }
        response = self.client.post(
            "/api/v1/imports/parcels", headers=AUTH,
            data={"source": "Numérisation Google Earth Pro", "id_field": "code", "name_field": "nom"},
            files={"file": ("../../evil.geojson", self.json_bytes(collection), "application/geo+json")},
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["created"], 1)
        saved = list((self.data_dir / "uploads" / "parcels").iterdir())
        self.assertEqual(len(saved), 1)
        self.assertNotIn("..", saved[0].name)
        self.assertEqual(self.client.get("/api/v1/parcels/P-1").json()["name"], "Colline A")

        bad = self.client.post(
            "/api/v1/imports/parcels", headers=AUTH, data={"source": "x" * 5},
            files={"file": ("script.exe", b"MZ", "application/octet-stream")},
        )
        self.assertEqual(bad.status_code, 422)

    def test_metrics_import_reports_invalid_rows(self):
        self.load_demo()
        csv = "id,ndvi,slope_pct,unknown_col\nparcelle_01,0.5,20,1\nparcelle_02,3.0,abc,1\nghost,0.1,10,1\n"
        response = self.client.post(
            "/api/v1/imports/metrics", headers=AUTH,
            data={"source": "Relevé 2026", "observed_start": "2026-01-01", "observed_end": "2026-06-30"},
            files={"file": ("m.csv", csv.encode(), "text/csv")},
        )
        report = response.json()
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(report["created"], 2)
        self.assertEqual(len(report["errors"]), 3)  # ndvi out of range, slope NaN, ghost parcel
        self.assertTrue(report["warnings"])
        metrics = self.client.get("/api/v1/parcels/parcelle_01/metrics").json()
        self.assertEqual(metrics["metrics"]["ndvi"], 0.5)  # dated import is the latest
        self.assertEqual(metrics["provenance"]["ndvi"]["source"], "Relevé 2026")

    def test_delete_parcel(self):
        self.load_demo()
        self.assertEqual(self.client.delete("/api/v1/parcels/parcelle_03").status_code, 401)
        self.assertEqual(self.client.delete("/api/v1/parcels/parcelle_03", headers=AUTH).status_code, 204)
        self.assertEqual(self.client.get("/api/v1/parcels/parcelle_03").status_code, 404)


class DecisionApiTests(ApiTestCase):
    def test_summary_matches_legacy_rule_engine_on_demo(self):
        self.load_demo()
        summary = self.client.get("/api/v1/decisions/summary").json()
        self.assertEqual(summary["by_strategy"], {"rna": 1, "reboisement": 1, "agroforesterie": 1})
        detail = self.client.get("/api/v1/decisions/parcels/parcelle_02").json()
        self.assertTrue(detail["based_on_demo_data"])
        self.assertTrue(detail["evaluation"]["legacy_rule_engine"]["agrees"])
        self.assertIn("validation", detail["evaluation"]["validation_note"])

    def test_evaluate_validates_inputs(self):
        ok = self.client.post("/api/v1/decisions/evaluate",
                              json={"metrics": {"slope_pct": 45, "ndvi": 0.15, "cropland": False}})
        self.assertEqual(ok.status_code, 200)
        self.assertEqual(ok.json()["recommended"]["strategy_id"], "reboisement")
        self.assertEqual(self.client.post("/api/v1/decisions/evaluate",
                                          json={"metrics": {"ndvi": 2}}).status_code, 422)
        self.assertEqual(self.client.post("/api/v1/decisions/evaluate",
                                          json={"metrics": {"hack": 1}}).status_code, 422)

    def test_custom_pairwise_and_ahp_endpoint(self):
        identity = [[1, 1, 1, 1]] * 4
        response = self.client.post("/api/v1/decisions/evaluate", json={
            "metrics": {"slope_pct": 25, "distance_to_forest_m": 300, "ndvi": 0.4, "cropland": False},
            "pairwise": {"rna": identity},
        })
        rna = next(r for r in response.json()["ranking"] if r["strategy_id"] == "rna")
        self.assertEqual([c["weight"] for c in rna["criteria"]], [0.25] * 4)
        self.assertEqual(self.client.post("/api/v1/decisions/ahp",
                                          json={"matrix": [[1, 3], [2, 1]]}).status_code, 422)
        bad_size = self.client.post("/api/v1/decisions/evaluate",
                                    json={"metrics": {}, "pairwise": {"rna": [[1]]}})
        self.assertEqual(bad_size.status_code, 422)

    def test_recommendation_history(self):
        self.load_demo()
        saved = self.client.post("/api/v1/decisions/parcels/parcelle_01", headers=AUTH)
        self.assertEqual(saved.status_code, 200, saved.text)
        history = self.client.get("/api/v1/decisions/parcels/parcelle_01/history").json()
        self.assertEqual(history[0]["strategy_id"], "rna")
        self.assertTrue(history[0]["based_on_demo_data"])

    def test_rules_endpoint_exposes_weights(self):
        rules = self.client.get("/api/v1/decisions/rules").json()
        self.assertTrue(all(c["ahp"]["consistent"] for c in rules["criteria"].values()))
        self.assertEqual(rules["rules"]["rna"]["forest_distance_max_m"], 800)


class GridRasterMixin:
    def _grid_raster(self, name: str, fill):
        grid = demo_grid()
        rows, cols = np.mgrid[0 : grid["height"], 0 : grid["width"]]
        values = fill(rows, cols).astype("float32")
        return write_raster(self.root / "inputs" / f"{name}.tif", values, left=grid["left"],
                            top=grid["top"], resolution=grid["resolution"])


class DatasetAndProcessingTests(GridRasterMixin, ApiTestCase):
    def test_raster_provenance_rules(self):
        path = self._grid_raster("dem", lambda r, c: 1500 + r * 0.0)
        sentinel = self.upload_raster(path, "s2", source="NASA", product="Sentinel-2 L2A")
        self.assertEqual(sentinel.status_code, 422)
        not_nasa = self.upload_raster(path, "esa", source="ESA Copernicus", product="DEM")
        self.assertEqual(not_nasa.status_code, 422)
        ok = self.upload_raster(path, "dem", variable="elevation_m")
        self.assertEqual(ok.status_code, 201, ok.text)
        self.assertEqual(ok.json()["crs"], "EPSG:32735")
        self.assertEqual(ok.json()["resolution_m"], 30.0)
        self.assertEqual(self.upload_raster(path, "dem").status_code, 422)  # duplicate name

    def test_terrain_zonal_statistics_feed_decisions(self):
        self.load_demo()
        # Plane rising 9 m per 30 m pixel northward -> slope 30 %.
        dem = self._grid_raster("dem", lambda r, c: 3000 - r * 9.0 + c * 0.0)
        dem_id = self.upload_raster(dem, "dem", variable="elevation_m").json()["id"]
        derived = self.client.post("/api/v1/processing/terrain", headers=AUTH, json={"dataset_id": dem_id})
        self.assertEqual(derived.status_code, 200, derived.text)
        slope_id = next(d["id"] for d in derived.json() if d["variable"] == "slope_pct")
        zonal = self.client.post("/api/v1/processing/zonal-statistics", headers=AUTH,
                                 json={"dataset_ids": [slope_id], "parcel_codes": ["parcelle_01"]})
        self.assertEqual(zonal.status_code, 200, zonal.text)
        stored = zonal.json()["stored"][0]
        self.assertAlmostEqual(stored["value"], 30.0, places=2)
        metrics = self.client.get("/api/v1/parcels/parcelle_01/metrics").json()
        self.assertEqual(metrics["provenance"]["slope_pct"]["method"], "zonal_mean")
        self.assertAlmostEqual(metrics["metrics"]["slope_pct"], 30.0, places=2)

    def test_vector_proximity(self):
        self.load_demo()
        forest = {"type": "FeatureCollection", "features": [{
            "type": "Feature", "properties": {},
            "geometry": {"type": "Point", "coordinates": [28.790, -2.3500]}}]}
        upload = self.client.post(
            "/api/v1/datasets/vector", headers=AUTH,
            data={"name": "forets", "source": "Inventaire local ICCN", "product": "Bosquets"},
            files={"file": ("forets.geojson", self.json_bytes(forest), "application/geo+json")},
        )
        self.assertEqual(upload.status_code, 201, upload.text)
        result = self.client.post("/api/v1/processing/proximity", headers=AUTH, json={
            "dataset_id": upload.json()["id"], "variable": "distance_to_forest_m",
            "parcel_codes": ["parcelle_01"]})
        self.assertEqual(result.status_code, 200, result.text)
        # Parcel north edge is at -2.352 -> about 221 m from the point.
        self.assertAlmostEqual(result.json()["stored"][0]["value"], 221, delta=5)
        bad = self.client.post("/api/v1/processing/proximity", headers=AUTH, json={
            "dataset_id": upload.json()["id"], "variable": "../../etc"})
        self.assertEqual(bad.status_code, 422)

    def test_pipeline_without_boundary_is_explicit(self):
        response = self.client.post("/api/v1/processing/pipeline", headers=AUTH)
        self.assertEqual(response.status_code, 422)
        self.assertIn("boundary", response.json()["detail"])


class ChangeAnalysisApiTests(GridRasterMixin, ApiTestCase):
    def test_raster_change_and_parcel_trend(self):
        self.load_demo()
        before = self._grid_raster("ndvi_2016", lambda r, c: 0.6 + r * 0.0)
        after = self._grid_raster("ndvi_2025", lambda r, c: np.where(c < 100, 0.3, 0.6) + r * 0.0)
        b = self.upload_raster(before, "ndvi_2016", product="HLSL30 NDVI", variable="ndvi",
                               period_start="2016-01-01", period_end="2016-12-31").json()["id"]
        a = self.upload_raster(after, "ndvi_2025", product="HLSL30 NDVI", variable="ndvi",
                               period_start="2025-01-01", period_end="2025-12-31").json()["id"]
        swapped = self.client.post("/api/v1/analysis/raster-change", headers=AUTH, json={
            "before_dataset_id": a, "after_dataset_id": b, "name": "x"})
        self.assertEqual(swapped.status_code, 422)
        response = self.client.post("/api/v1/analysis/raster-change", headers=AUTH, json={
            "before_dataset_id": b, "after_dataset_id": a, "threshold": 0.1, "name": "ndvi_16_25"})
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertGreater(body["classes"]["perte"]["pixels"], 0)
        self.assertEqual(body["classes"]["gain"]["pixels"], 0)
        self.assertIsNotNone(body["classes"]["perte"]["area_ha"])
        by_parcel = {p["parcel"]: p["mean_difference"] for p in body["per_parcel"]}
        self.assertAlmostEqual(by_parcel["parcelle_03"], -0.3, places=3)  # western parcel lost NDVI
        self.assertAlmostEqual(by_parcel["parcelle_02"], 0.0, places=3)
        self.assertEqual(len(self.client.get("/api/v1/analysis").json()), 1)

        for year, value in ((2019, 0.50), (2021, 0.45), (2023, 0.40), (2025, 0.33)):
            csv = f"id,ndvi\nparcelle_03,{value}\n"
            self.client.post("/api/v1/imports/metrics", headers=AUTH,
                             data={"source": "MODIS MOD13Q1", "observed_start": f"{year}-01-01",
                                   "observed_end": f"{year}-12-31"},
                             files={"file": ("n.csv", csv.encode(), "text/csv")})
        trend = self.client.get("/api/v1/analysis/parcels/parcelle_03/trend",
                                params={"variable": "ndvi"}).json()
        self.assertEqual(trend["status"], "tested")
        self.assertLess(trend["sens_slope_per_year"], 0)
        self.assertEqual(trend["undated_observations_ignored"], 1)  # demo CSV value has no date


class NasaApiTests(ApiTestCase):
    def _power_route(self):
        monthly = {
            "PRECTOTCORR": {"202401": 4.0, "202402": 6.0, "202413": 5.0, "202403": -999.0},
            "T2M": {"202401": 20.0, "202402": 22.0, "202413": 21.0, "202403": 21.0},
        }
        self.nasa.route("power.larc.nasa.gov/api/temporal/monthly",
                        lambda req: httpx.Response(200, json=power_payload(monthly, {"T2M": "C"})))

    def test_power_point_parses_and_logs(self):
        self._power_route()
        response = self.client.get("/api/v1/nasa/power/point", params={
            "latitude": -2.35, "longitude": 28.79, "start": "2024-01-01", "end": "2024-12-31",
            "parameters": "PRECTOTCORR,T2M"})
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["summary"]["PRECTOTCORR"], {"count": 2, "missing": 1, "mean": 5.0,
                                                          "min": 4.0, "max": 6.0})
        self.assertEqual(body["annual"]["T2M"], {"2024": 21.0})
        sent = self.nasa.calls[-1]
        self.assertEqual(sent.url.params["start"], "2024")
        self.assertEqual(sent.url.params["community"], "AG")
        with self.app.state.database.session() as session:
            self.assertEqual(session.query(NasaRequestLog).filter_by(service="power", ok=True).count(), 1)

    def test_power_input_validation(self):
        bad = self.client.get("/api/v1/nasa/power/point", params={
            "latitude": -2.35, "longitude": 28.79, "start": "2024-01-01", "end": "2024-12-31",
            "parameters": "T2M;DROP TABLE"})
        self.assertEqual(bad.status_code, 422)
        future = self.client.get("/api/v1/nasa/power/point", params={
            "latitude": -2.35, "longitude": 28.79, "start": "2024-01-01", "end": "2999-01-01"})
        self.assertEqual(future.status_code, 422)
        self.assertEqual(self.nasa.calls, [])

    def test_power_for_parcel_stores_observations(self):
        self.load_demo()
        self._power_route()
        response = self.client.post("/api/v1/nasa/power/parcels/parcelle_01", headers=AUTH,
                                    json={"start": "2024-01-01", "end": "2024-12-31"})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertAlmostEqual(response.json()["stored"]["rainfall_mm"], 5.0 * 365.25, places=1)
        lat = float(self.nasa.calls[-1].url.params["latitude"])
        self.assertAlmostEqual(lat, -2.3565, places=3)  # parcel centroid

    def test_upstream_errors_are_retried_then_reported(self):
        self.nasa.route("power.larc.nasa.gov", lambda req: httpx.Response(503, text="maintenance"))
        response = self.client.get("/api/v1/nasa/power/point", params={
            "latitude": -2.35, "longitude": 28.79, "start": "2024-01-01", "end": "2024-12-31"})
        self.assertEqual(response.status_code, 502)
        self.assertEqual(len(self.nasa.calls), 3)  # 1 try + 2 retries
        self.nasa.calls.clear()
        self.nasa.route("power.larc.nasa.gov", lambda req: httpx.Response(422, json={"messages": ["bad"]}))
        self.client.get("/api/v1/nasa/power/point", params={
            "latitude": -2.35, "longitude": 28.79, "start": "2024-01-01", "end": "2024-12-31"})
        self.assertEqual(len(self.nasa.calls), 1)  # client errors are not retried

    def test_cmr_granule_search_for_parcel(self):
        self.load_demo()
        feed = {"feed": {"entry": [{
            "id": "G1", "title": "HLS.L30.T35MQT.2025001", "time_start": "2025-01-01T08:00:00Z",
            "cloud_cover": "12", "links": [
                {"rel": "http://esipfed.org/ns/fedsearch/1.1/data#", "href": "https://data.lpdaac.earthdatacloud.nasa.gov/x/B04.tif"},
                {"rel": "http://esipfed.org/ns/fedsearch/1.1/browse#", "href": "https://x/browse.jpg"},
            ]}]}}
        self.nasa.route("cmr.earthdata.nasa.gov/search/granules.json", lambda req: httpx.Response(200, json=feed))
        response = self.client.get("/api/v1/nasa/cmr/granules", params={
            "product": "hls_landsat", "start": "2025-01-01", "end": "2025-03-31", "parcel_code": "parcelle_01"})
        self.assertEqual(response.status_code, 200, response.text)
        granule = response.json()["granules"][0]
        self.assertEqual(granule["data_links"], ["https://data.lpdaac.earthdatacloud.nasa.gov/x/B04.tif"])
        self.assertEqual(granule["cloud_cover"], 12.0)
        self.assertEqual(self.nasa.calls[-1].url.params["short_name"], "HLSL30")
        unknown = self.client.get("/api/v1/nasa/cmr/granules", params={
            "product": "hls_sentinel", "start": "2025-01-01", "end": "2025-03-31", "bbox": "28,-3,29,-2"})
        self.assertEqual(unknown.status_code, 404)

    def test_appeears_requires_credentials(self):
        self.load_demo()
        response = self.client.post("/api/v1/nasa/appeears/tasks", headers=AUTH, json={
            "task_name": "ndvi_kalehe", "product_key": "modis_ndvi",
            "start": "2020-01-01", "end": "2020-12-31"})
        self.assertEqual(response.status_code, 503)
        self.assertEqual(self.nasa.calls, [])


class AppeearsFlowTests(ApiTestCase):
    settings_overrides = {"earthdata_username": "user", "earthdata_password": "pass"}

    def test_submit_status_and_download(self):
        self.load_demo()
        task_id = "0f1e2d3c-4b5a-6978-8091-a2b3c4d5e6f7"
        file_id = "aaaabbbb-cccc-dddd-eeee-ffff00001111"
        submitted = {}

        def login(req):
            self.assertTrue(req.headers["Authorization"].startswith("Basic "))
            return httpx.Response(200, json={"token": "tok"})

        def task(req):
            self.assertEqual(req.headers["Authorization"], "Bearer tok")
            submitted.update(json.loads(req.content))
            return httpx.Response(202, json={"task_id": task_id, "status": "pending"})

        grid = demo_grid()
        tif = write_raster(self.root / "dl.tif", np.full((grid["height"], grid["width"]), 0.5),
                           left=grid["left"], top=grid["top"], resolution=grid["resolution"])
        self.nasa.route("appeears.earthdatacloud.nasa.gov/api/login", login)
        self.nasa.route("appeears.earthdatacloud.nasa.gov/api/task", task)
        self.nasa.route(f"appeears.earthdatacloud.nasa.gov/api/task/{task_id}",
                        lambda req: httpx.Response(200, json={"task_id": task_id, "status": "done"}))
        self.nasa.route(f"appeears.earthdatacloud.nasa.gov/api/bundle/{task_id}/{file_id}",
                        lambda req: httpx.Response(200, content=tif.read_bytes()))

        response = self.client.post("/api/v1/nasa/appeears/tasks", headers=AUTH, json={
            "task_name": "ndvi_kalehe", "product_key": "modis_ndvi",
            "start": "2020-01-01", "end": "2020-12-31", "parcel_codes": ["parcelle_01"]})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(submitted["task_name"], "ndvi_kalehe")
        self.assertEqual(submitted["params"]["layers"][0]["product"], "MOD13Q1.061")
        self.assertEqual(len(submitted["params"]["geo"]["features"]), 1)
        status = self.client.get(f"/api/v1/nasa/appeears/tasks/{task_id}", headers=AUTH)
        self.assertEqual(status.json()["status"], "done")

        download = self.client.post(f"/api/v1/nasa/appeears/tasks/{task_id}/download", headers=AUTH, json={
            "file_id": file_id, "file_name": "../MOD13Q1_NDVI.tif", "register_as": "modis_ndvi_2020",
            "variable": "ndvi", "product": "MOD13Q1.061", "period_start": "2020-01-01", "period_end": "2020-12-31"})
        self.assertEqual(download.status_code, 200, download.text)
        self.assertEqual(download.json()["path"], f"raw/nasa/appeears/{task_id}/MOD13Q1_NDVI.tif")
        dataset = self.client.get(f"/api/v1/datasets/{download.json()['dataset_id']}").json()
        self.assertEqual(dataset["source"], "NASA AppEEARS (LP DAAC)")
        logins = [c for c in self.nasa.calls if c.url.path.endswith("/login")]
        self.assertEqual(len(logins), 1)  # token reused


if __name__ == "__main__":
    unittest.main()
