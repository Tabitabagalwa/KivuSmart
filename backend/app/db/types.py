"""Dialect-aware geometry column.

On PostgreSQL the column is a PostGIS ``geometry(<type>, 4326)`` written with
``ST_GeomFromEWKT`` and read with ``ST_AsEWKB``. On other dialects (SQLite used by
the offline tests) the geometry is stored as WKT text. Python code always sees
Shapely geometries.
"""

from __future__ import annotations

from typing import Any

from shapely import wkb, wkt
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.sql.functions import GenericFunction
from sqlalchemy.types import Text, TypeDecorator

WGS84_SRID = 4326


class _geometry_from_ewkt(GenericFunction):
    inherit_cache = True
    name = "kivu_geometry_from_ewkt"


class _geometry_as_binary(GenericFunction):
    inherit_cache = True
    name = "kivu_geometry_as_binary"


@compiles(_geometry_from_ewkt)
def _from_ewkt_default(element, compiler, **kw):
    return compiler.process(element.clauses, **kw)


@compiles(_geometry_from_ewkt, "postgresql")
def _from_ewkt_postgresql(element, compiler, **kw):
    return f"ST_GeomFromEWKT({compiler.process(element.clauses, **kw)})"


@compiles(_geometry_as_binary)
def _as_binary_default(element, compiler, **kw):
    return compiler.process(element.clauses, **kw)


@compiles(_geometry_as_binary, "postgresql")
def _as_binary_postgresql(element, compiler, **kw):
    return f"ST_AsEWKB({compiler.process(element.clauses, **kw)})"


def to_shapely(value: Any) -> BaseGeometry:
    if isinstance(value, BaseGeometry):
        return value
    if isinstance(value, dict):
        return shape(value)
    if isinstance(value, str):
        text = value.split(";", 1)[1] if value.upper().startswith("SRID=") else value
        return wkt.loads(text)
    raise TypeError(f"Unsupported geometry value: {type(value).__name__}")


class GeometryType(TypeDecorator):
    impl = Text
    cache_ok = True

    def __init__(self, geometry_type: str = "GEOMETRY", srid: int = WGS84_SRID) -> None:
        super().__init__()
        self.geometry_type = geometry_type.upper()
        self.srid = srid

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            from geoalchemy2 import Geometry

            return dialect.type_descriptor(
                Geometry(geometry_type=self.geometry_type, srid=self.srid, spatial_index=False)
            )
        return dialect.type_descriptor(Text())

    def bind_expression(self, bindvalue):
        return _geometry_from_ewkt(bindvalue, type_=self)

    def column_expression(self, col):
        return _geometry_as_binary(col, type_=self)

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        geometry = to_shapely(value)
        if geometry.is_empty:
            raise ValueError("Empty geometries cannot be stored.")
        if dialect.name == "postgresql":
            return f"SRID={self.srid};{geometry.wkt}"
        return geometry.wkt

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        if isinstance(value, memoryview):
            value = value.tobytes()
        if isinstance(value, (bytes, bytearray)):
            return wkb.loads(bytes(value))
        if isinstance(value, str):
            stripped = value.strip()
            if stripped and all(c in "0123456789abcdefABCDEF" for c in stripped[:16]):
                return wkb.loads(stripped, hex=True)
            return to_shapely(stripped)
        if hasattr(value, "data"):  # geoalchemy2 WKBElement
            return wkb.loads(bytes(value.data))
        return to_shapely(value)
