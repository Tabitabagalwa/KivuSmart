"""Shared fixtures for backend API tests: isolated data dir, SQLite DB, fake NASA."""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any, Callable

import httpx
import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.transform import from_origin

from KivuSmart.backend.app.core.config import PROJECT_ROOT, Settings

API_KEY = "test-api-key-0123456789abcdef"
AUTH = {"X-API-Key": API_KEY}


def write_raster(
    path: Path,
    values: np.ndarray,
    *,
    left: float,
    top: float,
    resolution: float,
    crs: str = "EPSG:32735",
    nodata: float | None = -9999.0,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path, "w", driver="GTiff", width=values.shape[1], height=values.shape[0], count=1,
        dtype="float32", crs=crs, transform=from_origin(left, top, resolution, resolution),
        nodata=nodata,
    ) as dataset:
        dataset.write(values.astype("float32"), 1)
    return path


def demo_grid(resolution: float = 30.0) -> dict[str, float]:
    """UTM 35S grid covering the bundled pilot parcels with a margin."""
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32735", always_xy=True)
    xs, ys = transformer.transform([28.74, 28.83], [-2.39, -2.31])
    left, right = min(xs), max(xs)
    bottom, top = min(ys), max(ys)
    width = int((right - left) // resolution) + 1
    height = int((top - bottom) // resolution) + 1
    return {"left": left, "top": top, "width": width, "height": height, "resolution": resolution}


class FakeNasa:
    """Routes httpx requests to handlers keyed by host + path prefix."""

    def __init__(self) -> None:
        self.calls: list[httpx.Request] = []
        self.handlers: list[tuple[str, Callable[[httpx.Request], httpx.Response]]] = []

    def route(self, prefix: str, handler: Callable[[httpx.Request], httpx.Response]) -> None:
        self.handlers.insert(0, (prefix, handler))

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(request)
        target = f"{request.url.host}{request.url.path}"
        for prefix, handler in self.handlers:
            if target.startswith(prefix):
                return handler(request)
        return httpx.Response(404, json={"error": "no fake route"})


def power_payload(parameters: dict[str, dict[str, float]], units: dict[str, str] | None = None) -> dict[str, Any]:
    return {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [28.79, -2.35, 1500]},
        "header": {"fill_value": -999.0},
        "properties": {"parameter": parameters},
        "parameters": {k: {"units": (units or {}).get(k, "?")} for k in parameters},
    }


class ApiTestCase(unittest.TestCase):
    settings_overrides: dict[str, Any] = {}

    def setUp(self) -> None:
        from fastapi.testclient import TestClient

        from KivuSmart.backend.app.server import create_app

        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.root = Path(self._tmp.name)
        self.data_dir = self.root / "data"
        self.config_dir = self.root / "config"
        for relative in (
            "boundaries/sud_kivu_pilotes.geojson",
            "processed/parcelles_nasa_metrics.csv",
            "reference/referentiel_kivu.json",
        ):
            destination = self.data_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(PROJECT_ROOT / "data" / relative, destination)
        shutil.copytree(PROJECT_ROOT / "config", self.config_dir)

        options = {
            "environment": "test",
            "database_url": f"sqlite:///{(self.root / 'test.db').as_posix()}",
            "data_dir": self.data_dir,
            "config_dir": self.config_dir,
            "api_key": API_KEY,
            "auto_create_tables": True,
            "nasa_cache_ttl_hours": 0,
            "nasa_max_retries": 2,
            "rate_limit_per_minute": 10_000,
            "earthdata_username": None,
            "earthdata_password": None,
            "earthdata_token": None,
            **self.settings_overrides,
        }
        self.settings = Settings(_env_file=None, **options)
        self.nasa = FakeNasa()
        self.app = create_app(self.settings, nasa_transport=httpx.MockTransport(self.nasa))
        self.app.state.nasa_sleep = lambda _seconds: None
        self.client = TestClient(self.app)
        self.client.__enter__()

    def tearDown(self) -> None:
        self.client.__exit__(None, None, None)
        self._tmp.cleanup()

    # helpers ---------------------------------------------------------------
    def load_demo(self) -> dict:
        response = self.client.post("/api/v1/imports/demo", headers=AUTH)
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def upload_raster(self, path: Path, name: str, **form: Any) -> httpx.Response:
        data = {"name": name, "source": "NASA LP DAAC", "product": "NASADEM_HGT",
                "satellite": "true", **{k: str(v) for k, v in form.items()}}
        with path.open("rb") as handle:
            return self.client.post(
                "/api/v1/datasets/raster", headers=AUTH, data=data,
                files={"file": (path.name, handle, "image/tiff")},
            )

    @staticmethod
    def json_bytes(payload: Any) -> bytes:
        return json.dumps(payload).encode("utf-8")
