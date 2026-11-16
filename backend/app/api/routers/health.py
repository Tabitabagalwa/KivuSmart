from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from ...decision_engine import ENGINE_VERSION

router = APIRouter(tags=["santé"])


@router.get("/health", summary="Vivacité du service")
def health(request: Request) -> dict:
    settings = request.app.state.settings
    return {
        "status": "ok",
        "service": settings.app_name,
        "environment": settings.environment,
        "decision_engine_version": ENGINE_VERSION,
    }


@router.get("/health/ready", summary="Disponibilité de la base PostgreSQL/PostGIS")
def ready(request: Request):
    report = request.app.state.database.check()
    status_code = 200 if report["connected"] else 503
    return JSONResponse({"status": "ready" if report["connected"] else "unavailable", "database": report},
                        status_code=status_code)
