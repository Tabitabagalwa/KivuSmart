from __future__ import annotations

import tempfile
import unittest
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from KivuSmart.backend.app.processing.errors import GeospatialProcessingError
from KivuSmart.backend.app.processing.features import extract_zonal_statistics
from KivuSmart.backend.app.processing.indices.vegetation import calculate_ndvi
from KivuSmart.backend.app.processing.pipeline import (
    RasterLayerInput,
    run_configured_pipeline,
    run_geospatial_pipeline,
)
from KivuSmart.backend.app.processing.raster.harmonize import RasterLayer, harmonize_rasters
from KivuSmart.backend.app.processing.raster.terrain import derive_terrain_attributes
from KivuSmart.backend.app.processing.vector.boundaries import load_study_area
from KivuSmart.backend.app.processing.vector.clip import clip_vector_layer


class GeospatialProcessingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.boundary_path = self.root / "official_boundary.geojson"
        boundary = gpd.GeoDataFrame(
            {"admin_name": ["Kalehe"]},
            geometry=[box(500_012, 9_750_012, 500_088, 9_750_088)],
            crs="EPSG:32735",
        )
        boundary.to_file(self.boundary_path, driver="GeoJSON", index=False)
        self.boundary = load_study_area(
            self.boundary_path, boundary_source="Synthetic test geometry"
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _write_raster(
        self,
        path: Path,
        values: np.ndarray,
        pixel_size: float = 10.0,
        scale: float = 1.0,
        offset: float = 0.0,
    ) -> Path:
        profile = {
            "driver": "GTiff",
            "height": values.shape[0],
            "width": values.shape[1],
            "count": 1,
            "dtype": values.dtype,
            "crs": "EPSG:32735",
            "transform": from_origin(500_000, 9_750_100, pixel_size, pixel_size),
            "nodata": -9999,
        }
        with rasterio.open(path, "w", **profile) as dataset:
            dataset.write(values, 1)
            dataset.scales = (scale,)
            dataset.offsets = (offset,)
        return path

    def _write_geographic_raster(self, path: Path) -> Path:
        bounds = self.boundary.to_crs("EPSG:4326").total_bounds
        min_x, min_y, max_x, max_y = bounds
        padding = 0.0001
        width = height = 8
        profile = {
            "driver": "GTiff",
            "height": height,
            "width": width,
            "count": 1,
            "dtype": "float32",
            "crs": "EPSG:4326",
            "transform": from_origin(
                min_x - padding,
                max_y + padding,
                (max_x - min_x + 2 * padding) / width,
                (max_y - min_y + 2 * padding) / height,
            ),
            "nodata": -9999,
        }
        with rasterio.open(path, "w", **profile) as dataset:
            dataset.write(np.full((height, width), 11, dtype=np.float32), 1)
        return path

    def test_boundary_is_dissolved_and_projected(self) -> None:
        self.assertEqual(self.boundary.crs.to_epsg(), 32735)
        self.assertEqual(len(self.boundary), 1)
        self.assertAlmostEqual(self.boundary.geometry.iloc[0].area, 5776)

    def test_boundary_without_declared_crs_is_rejected(self) -> None:
        no_crs = self.root / "no_crs.shp"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            gpd.GeoDataFrame(
                geometry=[box(0, 0, 1, 1)], crs=None
            ).to_file(no_crs, index=False)
        projection_file = no_crs.with_suffix(".prj")
        if projection_file.exists():
            projection_file.unlink()
        with self.assertRaisesRegex(GeospatialProcessingError, "no declared CRS"):
            load_study_area(no_crs, boundary_source="Synthetic test geometry")

    def test_boundary_reprojects_from_geographic_crs(self) -> None:
        geographic = self.root / "boundary_wgs84.geojson"
        gpd.GeoDataFrame(
            {"admin_name": ["Kalehe"]},
            geometry=[self.boundary.to_crs("EPSG:4326").geometry.iloc[0]],
            crs="EPSG:4326",
        ).to_file(geographic, driver="GeoJSON", index=False)
        projected = load_study_area(
            geographic, boundary_source="Synthetic test geometry"
        )
        self.assertEqual(projected.crs.to_epsg(), 32735)
        self.assertLess(
            projected.geometry.iloc[0].hausdorff_distance(
                self.boundary.geometry.iloc[0]
            ),
            0.01,
        )

    def test_vector_layer_is_reprojected_and_clipped(self) -> None:
        vector_path = self.root / "source_vectors.geojson"
        source_geometry = gpd.GeoDataFrame(
            {"id": ["inside", "outside"]},
            geometry=[
                box(500_000, 9_750_000, 500_040, 9_750_040),
                box(500_200, 9_750_200, 500_240, 9_750_240),
            ],
            crs="EPSG:32735",
        ).to_crs("EPSG:4326")
        source_geometry.to_file(vector_path, driver="GeoJSON", index=False)
        clipped = clip_vector_layer(vector_path, self.boundary)
        self.assertEqual(clipped.crs.to_epsg(), 32735)
        self.assertEqual(clipped["id"].tolist(), ["inside"])
        self.assertTrue(
            self.boundary.geometry.iloc[0].covers(clipped.geometry.iloc[0])
        )

    def test_rasters_are_clipped_to_one_non_finer_grid(self) -> None:
        fine = self._write_raster(
            self.root / "fine.tif", np.full((10, 10), 3, dtype=np.float32), 10
        )
        coarse = self._write_raster(
            self.root / "coarse.tif", np.full((5, 5), 7, dtype=np.float32), 20
        )
        layers = [
            RasterLayer("fine", fine, "NASA", True, "NASA DEM"),
            RasterLayer("coarse", coarse, "NASA", True, "NASA rainfall"),
        ]
        outputs, report = harmonize_rasters(
            layers, self.boundary, self.root / "outputs"
        )
        self.assertEqual(report["target_resolution_m"], 20)
        self.assertEqual(report["layers"]["fine"]["native_resolution_m"], [10, 10])
        with rasterio.open(outputs["fine"]) as first, rasterio.open(
            outputs["coarse"]
        ) as second:
            self.assertEqual(first.crs, second.crs)
            self.assertEqual(first.transform, second.transform)
            self.assertEqual(first.shape, second.shape)
            self.assertEqual(abs(first.transform.a), 20)
            values = first.read(1, masked=True)
            self.assertTrue(np.ma.getmaskarray(values).any())
            self.assertTrue(np.allclose(values.compressed(), 3))

    def test_geographic_raster_is_reprojected_to_study_area_crs(self) -> None:
        raster = self._write_geographic_raster(self.root / "geographic.tif")
        outputs, report = harmonize_rasters(
            [RasterLayer("rainfall", raster, "NASA", True, "GPM IMERG")],
            self.boundary,
            self.root / "outputs",
        )
        with rasterio.open(outputs["rainfall"]) as prepared:
            self.assertEqual(prepared.crs.to_epsg(), 32735)
            self.assertGreater(prepared.read(1, masked=True).count(), 0)
            self.assertTrue(
                np.allclose(prepared.read(1, masked=True).compressed(), 11)
            )
        self.assertGreater(
            report["layers"]["rainfall"]["native_resolution_m"][0], 0
        )

    def test_categorical_raster_uses_nearest_neighbour(self) -> None:
        values = (
            np.indices((10, 10)).sum(axis=0).astype(np.float32) % 2 + 1
        )
        raster = self._write_raster(
            self.root / "land_cover.tif",
            values,
            scale=1,
        )
        outputs, report = harmonize_rasters(
            [
                RasterLayer(
                    "land_cover",
                    raster,
                    "Local authority",
                    False,
                    "Land-cover classes",
                    categorical=True,
                )
            ],
            self.boundary,
            self.root / "outputs",
        )
        with rasterio.open(outputs["land_cover"]) as prepared:
            classes = set(prepared.read(1, masked=True).compressed().tolist())
            self.assertLessEqual(classes, {1, 2})
        self.assertEqual(
            report["layers"]["land_cover"]["resampling"], "nearest"
        )

    def test_requested_finer_than_native_grid_is_rejected(self) -> None:
        raster = self._write_raster(
            self.root / "source.tif", np.ones((10, 10), dtype=np.float32), 10
        )
        with self.assertRaisesRegex(GeospatialProcessingError, "false detail"):
            harmonize_rasters(
                [RasterLayer("rain", raster, "NASA", True, "GPM")],
                self.boundary,
                self.root / "outputs",
                target_resolution_m=5,
            )

    def test_non_nasa_satellite_and_sentinel_inputs_are_rejected(self) -> None:
        raster = self._write_raster(
            self.root / "source.tif", np.ones((10, 10), dtype=np.float32), 10
        )
        for layer in (
            RasterLayer("other", raster, "Other agency", True, "product"),
            RasterLayer("bad", raster, "NASA", True, "Sentinel product"),
        ):
            with self.subTest(layer=layer.name):
                with self.assertRaises(GeospatialProcessingError):
                    harmonize_rasters(
                        [layer], self.boundary, self.root / "outputs"
                    )

    def test_ndvi_and_zonal_stats_are_measurable(self) -> None:
        red = self._write_raster(
            self.root / "red.tif",
            np.full((10, 10), 1000, dtype=np.float32),
            scale=0.001,
        )
        nir = self._write_raster(
            self.root / "nir.tif",
            np.full((10, 10), 3000, dtype=np.float32),
            scale=0.001,
        )
        ndvi_path = calculate_ndvi(
            red, nir, self.root / "ndvi.tif", "NASA", "VIIRS"
        )
        stats = extract_zonal_statistics({"ndvi": ndvi_path}, self.boundary)
        self.assertAlmostEqual(stats.iloc[0]["mean"], 0.5, places=6)
        self.assertGreater(stats.iloc[0]["valid_pixel_count"], 0)

    def test_terrain_derivatives_follow_known_plane(self) -> None:
        values = np.tile(np.arange(10, dtype=np.float32) * 10, (10, 1))
        dem = self._write_raster(self.root / "dem.tif", values, 10)
        outputs = derive_terrain_attributes(dem, self.root / "terrain")
        with rasterio.open(outputs["slope_percent"]) as slope:
            valid = slope.read(1, masked=True).compressed()
            self.assertTrue(np.allclose(valid, 100, atol=1e-4))
        with rasterio.open(outputs["aspect_degrees"]) as aspect:
            valid = aspect.read(1, masked=True).compressed()
            self.assertTrue(np.allclose(valid, 270, atol=1e-4))

    def test_pipeline_writes_verified_intermediate_outputs(self) -> None:
        elevation = self._write_raster(
            self.root / "elevation.tif",
            np.full((10, 10), 100, dtype=np.float32),
        )
        report = run_geospatial_pipeline(
            self.boundary_path,
            "Synthetic test geometry",
            [
                RasterLayerInput(
                    name="elevation",
                    path=str(elevation),
                    source="NASA",
                    satellite=True,
                    product="NASADEM",
                )
            ],
            self.root / "result",
        )
        self.assertEqual(report["boundary_feature_count"], 1)
        self.assertTrue(Path(report["boundary_output_path"]).is_file())
        self.assertTrue(Path(report["zonal_statistics_path"]).is_file())
        self.assertTrue((self.root / "result" / "processing_report.json").is_file())

    def test_configured_pipeline_refuses_missing_official_boundary(self) -> None:
        config = self.root / "study_area.json"
        manifest = self.root / "data_sources.json"
        config.write_text(
            '{"boundary_path": null, "target_crs": "EPSG:32735"}',
            encoding="utf-8",
        )
        manifest.write_text('{"rasters": []}', encoding="utf-8")
        with self.assertRaisesRegex(
            GeospatialProcessingError, "No authoritative boundary"
        ):
            run_configured_pipeline(config, manifest, self.root / "result")


if __name__ == "__main__":
    unittest.main()
