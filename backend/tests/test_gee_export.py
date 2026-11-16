"""Offline tests of the GEE downloader (Earth Engine is simulated)."""

from __future__ import annotations

import io
import tempfile
import unittest
import zipfile
from dataclasses import replace
from pathlib import Path

import httpx
import numpy as np
import rasterio

from KivuSmart.backend.app.services.data_ingestion import gee_export as gee
from KivuSmart.backend.tests.support import write_raster


class FakeImage:
    def __init__(self):
        self.params = None

    def toFloat(self):
        return self

    def unmask(self, *_args):
        return self

    def getDownloadURL(self, params):
        self.params = params
        return "https://earthengine.googleapis.com/v1/thumbnails/x:getPixels"


class GeeExportTests(unittest.TestCase):
    def test_plan_defaults_and_overrides(self):
        names = [j.name for j in gee.plan_jobs(["hls_ndvi", "nasadem", "gpm_rainfall"])]
        self.assertEqual(names, ["hls_ndvi_2016", "hls_ndvi_2025", "nasadem", "gpm_rainfall_2015_2024"])
        jobs = gee.plan_jobs(["modis_ndvi"], years=[2020])
        self.assertEqual((jobs[0].start, jobs[0].end), ("2020-01-01", "2020-12-31"))
        with self.assertRaises(ValueError):
            gee.plan_jobs(["sentinel2"])

    def test_only_nasa_sources(self):
        for product in gee.PRODUCTS.values():
            self.assertIn("nasa", product.source.casefold())
            self.assertNotIn("sentinel", product.collection.casefold())
            self.assertNotIn("S30", product.collection)

    def test_grid_is_aligned_and_identical_across_years(self):
        bounds = gee.load_area_bounds(gee.DEFAULT_AREA)
        grid = gee.utm_grid(bounds, 30)
        _, _, left, _, _, top = grid["crs_transform"]
        self.assertEqual((left % 30, top % 30), (0, 0))
        self.assertEqual(grid, gee.utm_grid(bounds, 30))
        self.assertEqual(grid["crs"], "EPSG:32735")

    def test_download_writes_tagged_geotiff_and_registers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = write_raster(root / "src.tif", np.full((5, 5), 0.4), left=700000, top=9740000,
                                  resolution=30, nodata=None)
            archive = io.BytesIO()
            with zipfile.ZipFile(archive, "w") as zipped:  # GEE may answer with a zip
                zipped.writestr("hls.ndvi.tif", source.read_bytes())

            image = FakeImage()
            product = replace(gee.PRODUCTS["hls_ndvi"], build=lambda ee, s, e: image)
            job = gee.Job(product, "hls_ndvi_2025", "2025-01-01", "2025-12-31")
            http = httpx.Client(transport=httpx.MockTransport(
                lambda req: httpx.Response(200, content=archive.getvalue())))
            record = gee.download_job(None, job, (28.78, -2.37, 28.80, -2.35), root / "out", http=http)

            self.assertEqual(record["status"], "downloaded")
            self.assertEqual(image.params["crs"], "EPSG:32735")
            self.assertEqual(image.params["format"], "GEO_TIFF")
            with rasterio.open(record["path"]) as dataset:
                self.assertEqual(dataset.nodata, gee.NODATA)
                self.assertEqual(dataset.tags()["product"], product.product)
                self.assertEqual(dataset.tags()["period_end"], "2025-12-31")

            sent = {}

            def api(req):
                sent["key"] = req.headers.get("X-API-Key")
                sent["body"] = req.content
                return httpx.Response(201, json={"id": 7})

            result = gee.register_in_api(Path(record["path"]), job, "http://api", "k",
                                         http=httpx.Client(transport=httpx.MockTransport(api)))
            self.assertEqual(result, {"registered": True, "dataset_id": 7})
            self.assertEqual(sent["key"], "k")
            self.assertIn(b"NASA LP DAAC via Google Earth Engine", sent["body"])

    def test_non_tiff_answer_is_rejected(self):
        with self.assertRaises(ValueError):
            gee._extract_tiff(b'{"error": "User memory limit exceeded"}')

    def test_dry_run_needs_no_earth_engine(self):
        self.assertEqual(gee.main(["--dry-run", "--products", "hls_ndvi", "smap"]), 0)


if __name__ == "__main__":
    unittest.main()
