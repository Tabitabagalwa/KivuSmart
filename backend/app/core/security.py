"""API key authentication, rate limiting, security headers and safe file paths."""

from __future__ import annotations

import hmac
import logging
import re
import threading
import time
import uuid
from collections import defaultdict, deque
from pathlib import Path

from fastapi import HTTPException, Request, status
from fastapi.security import APIKeyHeader
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse, Response

logger = logging.getLogger("kivu.security")

API_KEY_HEADER = "X-API-Key"
api_key_scheme = APIKeyHeader(name=API_KEY_HEADER, auto_error=False)

_DOCS_PATHS = ("/docs", "/redoc", "/openapi.json")


def verify_api_key(request: Request, provided: str | None) -> None:
    """Reject the request unless the configured API key is supplied.

    Without a configured key, writes are allowed only outside production so that
    local development stays frictionless; production settings refuse to start
    without a key.
    """
    settings = request.app.state.settings
    if settings.api_key is None:
        if settings.environment == "production":  # defensive; settings forbid it
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "API key not configured.")
        return
    expected = settings.api_key.get_secret_value().encode("utf-8")
    if not provided or not hmac.compare_digest(provided.encode("utf-8"), expected):
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            "Clé API invalide ou absente.",
            headers={"WWW-Authenticate": API_KEY_HEADER},
        )


class RateLimiter:
    """Thread-safe sliding-window limiter keyed by client address.

    In-memory: suitable for one process. Use a shared store (Redis, reverse proxy)
    when running several workers.
    """

    def __init__(self, limit_per_minute: int, window_s: float = 60.0) -> None:
        self.limit = limit_per_minute
        self.window_s = window_s
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def hit(self, key: str, now: float | None = None) -> float | None:
        """Register a hit; return seconds to wait if the limit is exceeded."""
        now = time.monotonic() if now is None else now
        with self._lock:
            hits = self._hits[key]
            while hits and now - hits[0] >= self.window_s:
                hits.popleft()
            if len(hits) >= self.limit:
                return max(0.0, self.window_s - (now - hits[0]))
            hits.append(now)
            if len(self._hits) > 10_000:  # bound memory under address churn
                for stale in [k for k, v in self._hits.items() if not v]:
                    del self._hits[stale]
            return None


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, limiter: RateLimiter, exempt_prefixes: tuple[str, ...] = ("/health",)):
        super().__init__(app)
        self.limiter = limiter
        self.exempt_prefixes = exempt_prefixes

    async def dispatch(self, request: Request, call_next):
        if not request.url.path.startswith(self.exempt_prefixes):
            client = request.client.host if request.client else "unknown"
            retry_after = self.limiter.hit(client)
            if retry_after is not None:
                return JSONResponse(
                    {"detail": "Trop de requêtes. Réessayez plus tard."},
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    headers={"Retry-After": str(int(retry_after) + 1)},
                )
        return await call_next(request)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault("Cross-Origin-Resource-Policy", "same-site")
        if not request.url.path.startswith(_DOCS_PATHS):
            response.headers.setdefault(
                "Content-Security-Policy", "default-src 'none'; frame-ancestors 'none'"
            )
            response.headers.setdefault("Cache-Control", "no-store")
        return response


class BodySizeLimitMiddleware(BaseHTTPMiddleware):
    """Reject requests whose declared Content-Length exceeds the upload limit."""

    def __init__(self, app, max_bytes: int):
        super().__init__(app)
        self.max_bytes = max_bytes

    async def dispatch(self, request: Request, call_next):
        declared = request.headers.get("content-length")
        if declared is not None:
            try:
                too_large = int(declared) > self.max_bytes
            except ValueError:
                return JSONResponse({"detail": "Content-Length invalide."}, status_code=400)
            if too_large:
                return JSONResponse(
                    {"detail": "Fichier ou requête trop volumineux."},
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                )
        return await call_next(request)


_SAFE_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


def safe_filename(original: str | None, default_stem: str = "upload") -> str:
    """Return a collision-free filename with no directory components."""
    name = Path(original or "").name  # strips any client-supplied directories
    name = _SAFE_CHARS.sub("_", name).strip("._") or default_stem
    stem, dot, suffix = name.rpartition(".")
    if not dot:
        stem, suffix = name, ""
    stem = stem[:80] or default_stem
    unique = uuid.uuid4().hex[:12]
    return f"{unique}_{stem}.{suffix.lower()}" if suffix else f"{unique}_{stem}"


class UnsafePathError(ValueError):
    """Raised when a path escapes its allowed root directory."""


def ensure_within(path: Path, root: Path) -> Path:
    """Resolve ``path`` and refuse it if it escapes ``root`` (path traversal)."""
    resolved = Path(path).resolve()
    root_resolved = Path(root).resolve()
    if resolved != root_resolved and root_resolved not in resolved.parents:
        raise UnsafePathError("Chemin de fichier non autorisé.")
    return resolved
