"""Integration test against a real PostgreSQL/PostGIS server.

Skipped unless ``KIVU_TEST_POSTGIS_URL`` is set, e.g. with ``docker compose up -d db``::

    $env:KIVU_TEST_POSTGIS_URL = "postgresql+psycopg://kivu:kivu@localhost:5432/kivu_smart_test"
    python -m unittest backend.tests.test_postgis_integration -v

The tables of the target database are dropped and recreated: use a test database.
"""

from __future__ import annotations

import os
import unittest

from shapely.geometry import MultiPolygon, box

from KivuSmart.backend.app.db.base import Base
from KivuSmart.backend.app.db.init_db import init_database
from KivuSmart.backend.app.db.models import Parcel
from KivuSmart.backend.app.db.repositories import list_parcels
from KivuSmart.backend.app.db.session import Database

POSTGIS_URL = os.getenv("KIVU_TEST_POSTGIS_URL")


@unittest.skipUnless(POSTGIS_URL, "KIVU_TEST_POSTGIS_URL non défini")
class PostgisIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.database = Database(POSTGIS_URL)
        init_database(self.database)
        Base.metadata.drop_all(self.database.engine)
        init_database(self.database)

    def tearDown(self) -> None:
        Base.metadata.drop_all(self.database.engine)
        self.database.dispose()

    def test_geometry_round_trip_and_spatial_filter(self):
        report = self.database.check()
        self.assertTrue(report["connected"])
        self.assertIn("postgis_version", report)
        shape = MultiPolygon([box(28.786, -2.361, 28.794, -2.352)])
        with self.database.session() as session:
            session.add(Parcel(code="p1", name="p1", geometry=shape, area_ha=88.6,
                               centroid_lat=-2.3565, centroid_lon=28.79, source="test", attributes={}))
        with self.database.session() as session:
            inside, total = list_parcels(session, bbox=(28.78, -2.37, 28.80, -2.35))
            outside, _ = list_parcels(session, bbox=(10, 10, 11, 11))
            self.assertEqual(total, 1)
            self.assertTrue(inside[0].geometry.equals(shape))
            self.assertEqual(outside, [])


if __name__ == "__main__":
    unittest.main()
