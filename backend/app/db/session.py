from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool


class Database:
    """Owns one SQLAlchemy engine and its session factory."""

    def __init__(self, url: str, echo: bool = False) -> None:
        self.url = url
        kwargs: dict[str, Any] = {"echo": echo, "future": True}
        if url.startswith("sqlite"):
            kwargs["connect_args"] = {"check_same_thread": False}
            if ":memory:" in url or url in {"sqlite://", "sqlite:///"}:
                kwargs["poolclass"] = StaticPool
        else:
            kwargs.update(pool_pre_ping=True, pool_size=5, max_overflow=10)
        self.engine: Engine = create_engine(url, **kwargs)
        if self.engine.dialect.name == "sqlite":
            event.listen(self.engine, "connect", _enable_sqlite_foreign_keys)
        self.session_factory = sessionmaker(
            bind=self.engine, autoflush=False, expire_on_commit=False
        )

    @property
    def is_postgis(self) -> bool:
        return self.engine.dialect.name == "postgresql"

    @contextmanager
    def session(self) -> Iterator[Session]:
        session = self.session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def check(self) -> dict[str, Any]:
        """Return connectivity and PostGIS information without raising."""
        report: dict[str, Any] = {"dialect": self.engine.dialect.name, "connected": False}
        try:
            with self.engine.connect() as connection:
                connection.execute(text("SELECT 1"))
                report["connected"] = True
                if self.is_postgis:
                    report["postgis_version"] = connection.execute(
                        text("SELECT PostGIS_Lib_Version()")
                    ).scalar_one()
        except Exception as exc:  # report, never leak credentials from the URL
            report["error"] = type(exc).__name__
        report["spatial_backend"] = "postgis" if self.is_postgis else "python (shapely)"
        return report

    def dispose(self) -> None:
        self.engine.dispose()


def _enable_sqlite_foreign_keys(dbapi_connection, _record) -> None:
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()
