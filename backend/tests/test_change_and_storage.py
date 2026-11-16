from __future__ import annotations

import tempfile
import unittest
from datetime import date
from pathlib import Path

import numpy as np
from shapely.geometry import MultiPolygon, box
from sqlalchemy import select
from sqlalchemy.dialects import postgresql

from KivuSmart.backend.app.analysis.change_detection import raster_change
from KivuSmart.backend.app.analysis.timeseries import analyze_series, climate_anomaly, mann_kendall
from KivuSmart.backend.app.db.base import Base
from KivuSmart.backend.app.db.models import Parcel
from KivuSmart.backend.app.db.session import Database
from KivuSmart.backend.app.processing.errors import GeospatialProcessingError
from KivuSmart.backend.tests.support import write_raster


class RasterChangeTests(unittest.TestCase):
    def test_difference_and_classes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            before = np.full((4, 4), 0.5)
            after = before.copy()
            after[0, :] = 0.2   # loss
            after[1, :] = 0.8   # gain
            after[3, 3] = -9999  # nodata
            b = write_raster(root / "b.tif", before, left=700000, top=9740000, resolution=30)
            a = write_raster(root / "a.tif", after, left=700000, top=9740000, resolution=30)
            result = raster_change(b, a, root / "out", threshold=0.1, name="t")
            self.assertEqual(result["valid_pixel_count"], 15)
            self.assertEqual(result["classes"]["perte"]["pixels"], 4)
            self.assertEqual(result["classes"]["gain"]["pixels"], 4)
            self.assertEqual(result["classes"]["stable"]["pixels"], 7)
            self.assertAlmostEqual(result["classes"]["perte"]["area_ha"], 4 * 900 / 10_000)
            self.assertTrue(Path(result["classes_path"]).is_file())

    def test_misaligned_grids_are_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            b = write_raster(root / "b.tif", np.ones((4, 4)), left=700000, top=9740000, resolution=30)
            a = write_raster(root / "a.tif", np.ones((4, 4)), left=700015, top=9740000, resolution=30)
            with self.assertRaises(GeospatialProcessingError):
                raster_change(b, a, root / "out", threshold=0.1)


class TimeSeriesTests(unittest.TestCase):
    def test_significant_decline(self):
        points = [(date(2010 + i, 6, 1), 0.6 - 0.02 * i) for i in range(12)]
        result = analyze_series(points)
        self.assertEqual(result["trend"], "baisse")
        self.assertAlmostEqual(result["sens_slope_per_year"], -0.02, places=3)
        self.assertLess(result["mann_kendall"]["p_value"], 0.001)

    def test_insufficient_and_flat_series(self):
        self.assertEqual(analyze_series([(date(2020, 1, 1), 0.5)])["status"], "insufficient_data")
        self.assertEqual(analyze_series([(date(2020, 1, 1), 0.5), (date(2021, 1, 1), 0.4)])["status"],
                         "change_only")
        flat = analyze_series([(date(2020 + i, 1, 1), 0.5) for i in range(6)])
        self.assertEqual(flat["trend"], "aucune tendance significative")
        self.assertEqual(mann_kendall([1, 1, 1])["s"], 0)

    def test_climate_anomaly(self):
        records = [{"period": f"{y}{m:02d}", "value": 4.0 if y < 2021 else 3.0}
                   for y in range(2015, 2025) for m in range(1, 13)]
        result = climate_anomaly(records, (2015, 2020), (2021, 2024))
        self.assertAlmostEqual(result["anomaly"], -1.0)
        self.assertAlmostEqual(result["relative_anomaly"], -0.25)


class GeometryStorageTests(unittest.TestCase):
    def test_sqlite_round_trip(self):
        database = Database("sqlite://")
        Base.metadata.create_all(database.engine)
        shape = MultiPolygon([box(28.78, -2.36, 28.79, -2.35)])
        with database.session() as session:
            session.add(Parcel(code="p", name="p", geometry=shape, area_ha=1, centroid_lat=0,
                               centroid_lon=0, source="test", attributes={}))
        with database.session() as session:
            stored = session.scalar(select(Parcel)).geometry
        self.assertTrue(stored.equals(shape))
        database.dispose()

    def test_postgis_sql_uses_spatial_functions(self):
        dialect = postgresql.psycopg.dialect()
        select_sql = str(select(Parcel.geometry).compile(dialect=dialect))
        self.assertIn("ST_AsEWKB(parcels.geometry)", select_sql)
        insert_sql = str(Parcel.__table__.insert().values(geometry="x").compile(dialect=dialect))
        self.assertIn("ST_GeomFromEWKT(", insert_sql)
        bind = Parcel.__table__.c.geometry.type.process_bind_param(box(0, 0, 1, 1), dialect)
        self.assertTrue(bind.startswith("SRID=4326;POLYGON"))


if __name__ == "__main__":
    unittest.main()
