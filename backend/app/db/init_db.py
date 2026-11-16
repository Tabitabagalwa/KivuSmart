"""Create the PostGIS extension, tables and spatial indexes.

Usage (from the project root)::

    python -m backend.app.db.init_db            # create schema
    python -m backend.app.db.init_db --demo     # also load the bundled demo parcels
"""

from __future__ import annotations

import argparse

from sqlalchemy import text

from . import models  # noqa: F401  (registers tables on Base.metadata)
from .base import Base
from .session import Database

SPATIAL_INDEXES = {
    "ix_parcels_geometry_gist": ("parcels", "geometry"),
    "ix_study_areas_geometry_gist": ("study_areas", "geometry"),
}


def init_database(database: Database) -> None:
    if database.is_postgis:
        with database.engine.begin() as connection:
            connection.execute(text("CREATE EXTENSION IF NOT EXISTS postgis"))
    Base.metadata.create_all(database.engine)
    if database.is_postgis:
        with database.engine.begin() as connection:
            for index_name, (table, column) in SPATIAL_INDEXES.items():
                connection.execute(
                    text(
                        f"CREATE INDEX IF NOT EXISTS {index_name} "
                        f"ON {table} USING GIST ({column})"
                    )
                )


def main() -> None:
    from ..core.config import get_settings
    from ..services.data_ingestion.importers import load_demo_data

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true", help="load bundled demonstration data")
    args = parser.parse_args()

    settings = get_settings()
    database = Database(settings.database_url, echo=settings.database_echo)
    init_database(database)
    print(f"Schema ready ({database.check()})")
    if args.demo:
        with database.session() as session:
            report = load_demo_data(session, settings)
        print(f"Demo data loaded: {report}")
    database.dispose()


if __name__ == "__main__":
    main()
