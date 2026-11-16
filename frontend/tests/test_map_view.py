from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from KivuSmart.frontend import map_view


class MapViewTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.addCleanup(self.temporary_directory.cleanup)
        self.project_root_patch = patch.object(map_view, "PROJECT_ROOT", self.root)
        self.project_root_patch.start()
        self.addCleanup(self.project_root_patch.stop)

    def test_satellite_layer_must_be_nasa_and_not_sentinel(self) -> None:
        with self.assertRaisesRegex(ValueError, "source NASA"):
            map_view._validate_satellite_layer(
                {"satellite": True, "source": "Other", "product": "image"},
                "external",
            )
        with self.assertRaisesRegex(ValueError, "Sentinel"):
            map_view._validate_satellite_layer(
                {"satellite": False, "source": "Local", "product": "Sentinel-derived"},
                "image",
            )
        with self.assertRaisesRegex(ValueError, "explicitement"):
            map_view._validate_satellite_layer(
                {"source": "Local data"},
                "missing metadata",
            )

    def test_vector_layers_are_loaded_with_role_and_recommendation(self) -> None:
        path = self.root / "zones.geojson"
        gpd.GeoDataFrame(
            {
                "id": ["zone-1"],
                "recommendation": ["RNA"],
                "justification": ["Documented test"],
            },
            geometry=[box(500_000, 9_750_000, 500_100, 9_750_100)],
            crs="EPSG:32735",
        ).to_file(path, driver="GeoJSON", index=False)
        layers, issues = map_view._load_vector_config(
            {
                "layers": [
                    {
                        "name": "RNA test",
                        "path": str(path),
                        "role": "rna",
                        "source": "Local test",
                        "satellite": False,
                    }
                ]
            }
        )
        self.assertEqual(issues, [])
        self.assertEqual(len(layers), 1)
        feature = layers[0]["features"][0]
        self.assertEqual(feature["properties"]["_recommendation"], "RNA")
        self.assertEqual(feature["properties"]["_role"], "rna")
        x, y = feature["geometry"]["coordinates"][0][0]
        self.assertGreater(x, 20)
        self.assertLess(x, 40)
        self.assertGreater(y, -10)
        self.assertLess(y, 0)

    def test_missing_satellite_provenance_layer_is_not_loaded(self) -> None:
        path = self.root / "zones.geojson"
        gpd.GeoDataFrame(
            {"id": ["zone-1"]},
            geometry=[box(500_000, 9_750_000, 500_100, 9_750_100)],
            crs="EPSG:32735",
        ).to_file(path, driver="GeoJSON", index=False)
        layers, issues = map_view._load_vector_config(
            {
                "layers": [
                    {
                        "name": "Satellite-derived zones",
                        "path": str(path),
                        "role": "environmental",
                        "satellite": True,
                        "source": "Other provider",
                        "product": "Provider imagery",
                    }
                ]
            }
        )
        self.assertEqual(layers, [])
        self.assertIn("source NASA", issues[0])

    def test_pilot_demo_data_is_not_treated_as_a_recommendation(self) -> None:
        pilot_path = self.root / "pilots.geojson"
        metrics_path = self.root / "metrics.csv"
        gpd.GeoDataFrame(
            {"id": ["pilot-1"], "name": ["Pilot parcel"]},
            geometry=[box(500_000, 9_750_000, 500_100, 9_750_100)],
            crs="EPSG:32735",
        ).to_file(pilot_path, driver="GeoJSON", index=False)
        metrics_path.write_text(
            "id,name,slope_pct\npilot-1,Pilot parcel,15\n",
            encoding="utf-8",
        )
        with (
            patch.object(map_view, "PILOT_PARCELS", pilot_path),
            patch.object(map_view, "SAMPLE_METRICS", metrics_path),
        ):
            layer = map_view._load_pilot_layer(map_view._read_metrics())
        properties = layer["features"][0]["properties"]
        self.assertEqual(properties["_role"], "pilot")
        self.assertEqual(properties["_recommendation"], "")
        self.assertEqual(properties["metrique_slope_pct"], 15)
        self.assertNotIn("metrique_name", properties)

    def test_legend_only_shows_layers_that_are_available(self) -> None:
        map_object = map_view.folium.Map(location=[0, 0])
        map_view._add_legend(
            map_object,
            [{"role": "pilot"}],
            [],
        )
        rendered = map_object.get_root().render()
        self.assertIn("Parcelles pilotes (démonstration)", rendered)
        self.assertNotIn("Zone RNA documentée", rendered)

    def test_continuous_raster_is_renderable_with_real_bounds(self) -> None:
        path = self.root / "continuous.tif"
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            width=4,
            height=4,
            count=1,
            dtype="float32",
            crs="EPSG:32735",
            transform=from_origin(500_000, 9_750_040, 10, 10),
            nodata=-9999,
        ) as dataset:
            dataset.write(np.arange(16, dtype=np.float32).reshape(4, 4), 1)
        rgba, bounds, legend = map_view._raster_overlay(
            {"name": "elevation", "path": str(path), "categorical": False}
        )
        self.assertEqual(rgba.shape[2], 4)
        self.assertEqual(rgba.dtype, np.uint8)
        self.assertEqual(len(bounds), 2)
        self.assertEqual(len(legend), 2)
        self.assertGreater(bounds[1][0], bounds[0][0])

    def test_raster_overlay_is_clipped_to_study_boundary(self) -> None:
        path = self.root / "continuous.tif"
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            width=4,
            height=4,
            count=1,
            dtype="float32",
            crs="EPSG:32735",
            transform=from_origin(500_000, 9_750_040, 10, 10),
            nodata=-9999,
        ) as dataset:
            dataset.write(np.arange(16, dtype=np.float32).reshape(4, 4), 1)
        study_area = gpd.GeoSeries(
            [
                box(500_000, 9_750_000, 500_040, 9_750_040).difference(
                    box(500_000, 9_750_000, 500_020, 9_750_020)
                )
            ],
            crs="EPSG:32735",
        ).to_crs("EPSG:4326").iloc[0]
        boundary = {
            "features": [{"geometry": study_area.__geo_interface__}]
        }

        rgba, _, _ = map_view._raster_overlay(
            {"name": "elevation", "path": str(path), "categorical": False},
            boundary,
        )

        self.assertGreater(np.count_nonzero(rgba[:, :, 3]), 0)
        self.assertLess(np.count_nonzero(rgba[:, :, 3]), rgba.shape[0] * rgba.shape[1])

    def test_selected_zone_shows_environmental_raster_statistics(self) -> None:
        path = self.root / "environment.tif"
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            width=4,
            height=4,
            count=1,
            dtype="float32",
            crs="EPSG:32735",
            transform=from_origin(500_000, 9_750_040, 10, 10),
            nodata=-9999,
        ) as dataset:
            dataset.write(np.full((4, 4), 12, dtype=np.float32), 1)
            dataset.set_band_unit(1, "test-unit")
        zone_wgs84 = gpd.GeoSeries(
            [box(500_000, 9_750_000, 500_040, 9_750_040)],
            crs="EPSG:32735",
        ).to_crs("EPSG:4326").iloc[0]
        stats, issues = map_view._selected_raster_statistics(
            {
                "geometry": zone_wgs84.__geo_interface__,
                "properties": {"_map_id": "test::zone"},
            },
            [{"name": "environment", "path": str(path)}],
        )
        self.assertEqual(issues, [])
        self.assertEqual(stats[0]["mean"], 12)
        self.assertEqual(stats[0]["unit"], "test-unit")
        self.assertGreater(stats[0]["valid_pixel_count"], 0)

    def test_categories_require_explicit_colors(self) -> None:
        path = self.root / "classes.tif"
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            width=2,
            height=2,
            count=1,
            dtype="float32",
            crs="EPSG:32735",
            transform=from_origin(500_000, 9_750_020, 10, 10),
        ) as dataset:
            dataset.write(np.array([[1, 2], [1, 2]], dtype=np.float32), 1)
        with self.assertRaisesRegex(ValueError, "class_colors"):
            map_view._raster_overlay(
                {"name": "classes", "path": str(path), "categorical": True}
            )
        display, _, legend = map_view._raster_overlay(
            {
                "name": "classes",
                "path": str(path),
                "categorical": True,
                "class_colors": {"1": "#00ff00", "2": "#0000ff"},
            }
        )
        self.assertEqual(set(legend), {"1", "2"})
        self.assertTrue(np.all(display[:, :, 3] > 0))


if __name__ == "__main__":
    unittest.main()
