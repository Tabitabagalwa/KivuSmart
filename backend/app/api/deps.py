from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from typing import Any

from fastapi import Request, Security
from sqlalchemy.orm import Session

from ..core.config import Settings
from ..core.security import api_key_scheme, verify_api_key
from ..db.models import NasaRequestLog
from ..decision_engine.rules import load_rules
from ..services.nasa.client import NasaHttpClient


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_db(request: Request) -> Iterator[Session]:
    """Yield a session; routers commit explicitly, failures are rolled back."""
    session = request.app.state.database.session_factory()
    try:
        yield session
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def require_api_key(request: Request, api_key: str | None = Security(api_key_scheme)) -> None:
    verify_api_key(request, api_key)


def get_rules(request: Request) -> dict[str, Any]:
    settings: Settings = request.app.state.settings
    return load_rules(settings.config_dir, settings.data_dir / "reference" / "referentiel_kivu.json")


def nasa_client_factory(request: Request) -> Callable[[str], NasaHttpClient]:
    """Build NASA clients that log every call to ``nasa_request_logs``."""
    app = request.app
    settings: Settings = app.state.settings

    def audit(entry: dict[str, Any]) -> None:
        with app.state.database.session() as session:
            session.add(
                NasaRequestLog(
                    service=entry["service"],
                    url=str(entry.get("url", ""))[:2000],
                    status_code=entry.get("status_code"),
                    ok=bool(entry.get("ok")),
                    cached=bool(entry.get("cached", False)),
                    duration_ms=entry.get("duration_ms"),
                    error=(entry.get("error") or None) and str(entry["error"])[:2000],
                )
            )

    def factory(service: str) -> NasaHttpClient:
        token = settings.earthdata_token.get_secret_value() if settings.earthdata_token else None
        return NasaHttpClient(
            service,
            timeout_s=settings.nasa_timeout_s,
            max_retries=settings.nasa_max_retries,
            cache_dir=settings.nasa_cache_dir,
            cache_ttl_s=settings.nasa_cache_ttl_hours * 3600,
            bearer_token=token,
            transport=getattr(app.state, "nasa_transport", None),
            audit=audit,
            sleep=getattr(app.state, "nasa_sleep", None) or time.sleep,
        )

    return factory
