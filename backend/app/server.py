"""FastAPI application for the Kivu Smart restoration backend.

Run from the project root::

    uvicorn backend.app.server:app --reload

The Streamlit dashboards (``app.py`` and ``backend/app/main.py``) are unchanged.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .api.routers import processing
from .api.routers import analysis
from .api.routers import datasets
from .api.routers import decisions
from .api.routers import health
from .api.routers import imports
from .api.routers import nasa
from .api.routers import parcels
from .core.config import Settings, get_settings
from .core.security import (
    API_KEY_HEADER,
    BodySizeLimitMiddleware,
    RateLimiter,
    RateLimitMiddleware,
    SecurityHeadersMiddleware,
    UnsafePathError,
)
from .db.init_db import init_database
from .db.repositories import NotFoundError
from .db.session import Database
from .processing.errors import GeospatialProcessingError
from .services.data_ingestion.importers import ImportValidationError
from .services.nasa.errors import NasaApiError, NasaConfigurationError

logger = logging.getLogger("kivu.api")
API_PREFIX = "/api/v1"


def _error(status_code: int, detail: str, **extra) -> JSONResponse:
    return JSONResponse({"detail": detail, **extra}, status_code=status_code)


def _install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(NotFoundError)
    async def not_found(_: Request, exc: NotFoundError):
        return _error(404, str(exc))

    @app.exception_handler(GeospatialProcessingError)
    async def geospatial(_: Request, exc: GeospatialProcessingError):
        return _error(422, str(exc), type="geospatial_processing_error")

    @app.exception_handler(ImportValidationError)
    async def import_error(_: Request, exc: ImportValidationError):
        return _error(422, str(exc), type="import_validation_error")

    @app.exception_handler(UnsafePathError)
    async def unsafe_path(_: Request, exc: UnsafePathError):
        return _error(400, str(exc))

    @app.exception_handler(NasaConfigurationError)
    async def nasa_config(_: Request, exc: NasaConfigurationError):
        return _error(503, str(exc), type="nasa_configuration_error")

    @app.exception_handler(NasaApiError)
    async def nasa_error(request: Request, exc: NasaApiError):
        if exc.service == "appeears" and exc.status_code == 401:
            request.app.state.appeears_token = None  # force a fresh login next time
        return _error(502, str(exc), type="nasa_api_error", upstream_status=exc.status_code)

    @app.exception_handler(IntegrityError)
    async def integrity(_: Request, exc: IntegrityError):
        logger.warning("Integrity error: %s", exc.orig)
        return _error(409, "Conflit avec des données existantes (doublon ou référence invalide).")

    @app.exception_handler(SQLAlchemyError)
    async def database_error(_: Request, exc: SQLAlchemyError):
        logger.exception("Database error")
        return _error(503, "Base de données indisponible ou requête invalide.")

    @app.exception_handler(ValueError)
    async def value_error(_: Request, exc: ValueError):
        return _error(422, str(exc), type="validation_error")

    @app.exception_handler(Exception)
    async def unexpected(_: Request, exc: Exception):
        logger.exception("Unhandled error")  # details stay in server logs
        return _error(500, "Erreur interne du serveur.")

    @app.exception_handler(RequestValidationError)
    async def request_validation(_: Request, exc: RequestValidationError):
        errors = [
            {"loc": list(e.get("loc", [])), "msg": e.get("msg"), "type": e.get("type")}
            for e in exc.errors()
        ]
        return _error(422, "Requête invalide.", errors=errors)


def create_app(
    settings: Settings | None = None,
    *,
    database: Database | None = None,
    nasa_transport: httpx.BaseTransport | None = None,
) -> FastAPI:
    settings = settings or get_settings()
    database = database or Database(settings.database_url, echo=settings.database_echo)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        settings.ensure_directories()
        if settings.auto_create_tables:
            init_database(database)
        if settings.api_key is None:
            logger.warning("KIVU_API_KEY absent : écritures ouvertes (développement uniquement).")
        yield
        database.dispose()

    docs = settings.docs_enabled
    app = FastAPI(
        title=settings.app_name,
        version="2.0.0",
        description=(
            "API de restauration des paysages du Sud-Kivu : données NASA, traitement "
            "géospatial, analyse des changements et aide à la décision (RNA, reboisement, "
            "agroforesterie). Les recommandations exigent une validation de terrain."
        ),
        lifespan=lifespan,
        docs_url="/docs" if docs else None,
        redoc_url="/redoc" if docs else None,
        openapi_url="/openapi.json" if docs else None,
    )
    app.state.settings = settings
    app.state.database = database
    app.state.nasa_transport = nasa_transport
    app.state.appeears_token = None

    # Middleware order: the last added runs first.
    app.add_middleware(GZipMiddleware, minimum_size=1024)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type", API_KEY_HEADER],
    )
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(RateLimitMiddleware, limiter=RateLimiter(settings.rate_limit_per_minute))
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=settings.max_upload_bytes + 64 * 1024)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.allowed_hosts)

    _install_error_handlers(app)
    app.include_router(health.router)
    for module in (parcels, imports, datasets, nasa, processing, analysis, decisions):
        app.include_router(module.router, prefix=API_PREFIX)
    return app


def __getattr__(name: str):
    # ``backend.app.server:app`` is built on first access (uvicorn), so importing
    # this module in tests does not read the environment or open connections.
    if name == "app":
        instance = create_app()
        globals()["app"] = instance
        return instance
    raise AttributeError(name)
