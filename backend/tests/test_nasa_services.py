from __future__ import annotations

import tempfile
import unittest
from datetime import date
from pathlib import Path

from KivuSmart.backend.app.services.nasa import appeears, cmr
import httpx

from KivuSmart.backend.app.services.nasa import power
from KivuSmart.backend.app.services.nasa.catalog import NASA_PRODUCTS, NasaProduct
from KivuSmart.backend.app.services.nasa.client import NasaHttpClient, check_url_allowed
from KivuSmart.backend.app.services.nasa.errors import NasaApiError, NasaConfigurationError
from KivuSmart.backend.tests.support import power_payload


def make_client(handler, **kwargs) -> NasaHttpClient:
    return NasaHttpClient("test", transport=httpx.MockTransport(handler), sleep=lambda _s: None, **kwargs)


class ClientSecurityTests(unittest.TestCase):
    def test_only_https_nasa_hosts_are_allowed(self):
        check_url_allowed("https://power.larc.nasa.gov/api/x")
        for url in ("http://power.larc.nasa.gov/api", "https://evil.com/nasa.gov",
                    "https://nasa.gov.evil.com/", "file:///etc/passwd", "https://169.254.169.254/"):
            with self.subTest(url=url), self.assertRaises(NasaApiError):
                check_url_allowed(url)

    def test_bearer_token_only_when_requested(self):
        seen = []
        client = make_client(lambda r: seen.append(r.headers.get("Authorization")) or httpx.Response(200, json={}),
                             bearer_token="secret")
        client.get_json("https://cmr.earthdata.nasa.gov/a", use_cache=False)
        client.get_json("https://cmr.earthdata.nasa.gov/b", auth=True, use_cache=False)
        self.assertEqual(seen, [None, "Bearer secret"])

    def test_cache_avoids_repeated_calls(self):
        calls = []
        with tempfile.TemporaryDirectory() as tmp:
            client = make_client(lambda r: calls.append(r) or httpx.Response(200, json={"v": 1}),
                                 cache_dir=Path(tmp), cache_ttl_s=3600)
            first = client.get_json("https://power.larc.nasa.gov/x", params={"a": 1})
            second = client.get_json("https://power.larc.nasa.gov/x", params={"a": 1})
            client.close()
        self.assertEqual(first, second)
        self.assertEqual(len(calls), 1)

    def test_download_enforces_size_limit(self):
        client = make_client(lambda r: httpx.Response(200, content=b"x" * 5000))
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "f.bin"
            with self.assertRaises(NasaApiError):
                client.download("https://data.lpdaac.earthdatacloud.nasa.gov/f", target, max_bytes=1000)
            self.assertFalse(target.exists())
            self.assertFalse(target.with_name("f.bin.part").exists())
            client.download("https://data.lpdaac.earthdatacloud.nasa.gov/f", target, max_bytes=10_000)
            self.assertEqual(target.stat().st_size, 5000)
        client.close()


class PowerTests(unittest.TestCase):
    def test_request_validation(self):
        base = dict(latitude=-2.3, longitude=28.8, start=date(2020, 1, 1), end=date(2020, 12, 31))
        power.PowerRequest(**base).validate()
        for change in ({"latitude": 95}, {"start": date(1970, 1, 1)}, {"parameters": ("bad-name",)},
                       {"parameters": tuple(f"P{i}" for i in range(20))}, {"community": "XX"},
                       {"start": date(2021, 1, 1)}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                power.PowerRequest(**{**base, **change}).validate()

    def test_daily_query_uses_full_dates(self):
        request = power.PowerRequest(-2.3, 28.8, date(2020, 1, 1), date(2020, 1, 31), "daily")
        self.assertEqual((request.query()["start"], request.query()["end"]), ("20200101", "20200131"))

    def test_parse_climatology_and_metrics(self):
        values = {m: 3.0 for m in ("01", "02", "03")}
        payload = power_payload({"PRECTOTCORR": {**values, "ANN": 3.0}, "T2M": {"01": 19.5, "ANN": 19.5}})
        result = power.parse_power_response(payload, "climatology")
        self.assertEqual(result["annual"]["PRECTOTCORR"], {"ANN": 3.0})
        self.assertEqual(result["summary"]["PRECTOTCORR"]["count"], 3)
        metrics = power.to_parcel_metrics(result)
        self.assertEqual(metrics["rainfall_mm"], (round(3.0 * 365.25, 1), "mm/an"))
        self.assertEqual(metrics["temperature_c"][0], 19.5)

    def test_unexpected_payload(self):
        with self.assertRaises(NasaApiError):
            power.parse_power_response({"messages": ["oops"]}, "monthly")


class CmrAndAppeearsTests(unittest.TestCase):
    def test_sentinel_products_are_refused(self):
        sentinel = NasaProduct("s", "HLSS30", "2.0", "LP DAAC", ("ndvi",), "30 m", "x")
        with self.assertRaises(ValueError):
            cmr.search_granules(make_client(lambda r: httpx.Response(200)), sentinel,
                                (28, -3, 29, -2), date(2024, 1, 1), date(2024, 2, 1))
        self.assertNotIn("HLSS30", {p.short_name for p in NASA_PRODUCTS.values()})
        with self.assertRaises(ValueError):
            appeears.build_area_task("t", {"type": "FeatureCollection", "features": [{}]},
                                     [{"product": "Sentinel2", "layer": "B04"}], date(2024, 1, 1), date(2024, 2, 1))

    def test_bbox_validation(self):
        with self.assertRaises(ValueError):
            cmr.search_granules(make_client(lambda r: httpx.Response(200)), NASA_PRODUCTS["nasadem"],
                                (29, -2, 28, -3), date(2024, 1, 1), date(2024, 2, 1))

    def test_appeears_login_requires_credentials_and_ids_are_checked(self):
        with self.assertRaises(NasaConfigurationError):
            appeears.login(make_client(lambda r: httpx.Response(200)), None, None)
        with self.assertRaises(ValueError):
            appeears.task_status(make_client(lambda r: httpx.Response(200)), "../../login")


if __name__ == "__main__":
    unittest.main()
