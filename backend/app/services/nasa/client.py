"""HTTP client shared by NASA integrations.

Security and robustness rules:

* only ``https`` URLs on ``*.nasa.gov`` hosts are contacted (no SSRF to other hosts);
* credentials are attached only to NASA hosts; httpx drops the ``Authorization``
  header on cross-origin redirects (for example to pre-signed S3 URLs);
* transient failures (429, 5xx, network errors) are retried with backoff;
* JSON responses can be cached on disk with a TTL to spare NASA services.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

from .errors import NasaApiError

logger = logging.getLogger("kivu.nasa")

ALLOWED_HOST_SUFFIX = ".nasa.gov"
RETRY_STATUS = {429, 500, 502, 503, 504}
USER_AGENT = "KivuSmartRestoration/2.0 (+NASA Space Apps research prototype)"

AuditHook = Callable[[dict[str, Any]], None]


def check_url_allowed(url: str) -> None:
    parts = urlsplit(url)
    host = (parts.hostname or "").lower()
    if parts.scheme != "https" or not (
        host.endswith(ALLOWED_HOST_SUFFIX) or host == ALLOWED_HOST_SUFFIX[1:]
    ):
        raise NasaApiError("http", f"URL refusée (hôte non NASA ou non HTTPS) : {host or url}")


class NasaHttpClient:
    def __init__(
        self,
        service: str,
        *,
        timeout_s: float = 60.0,
        max_retries: int = 3,
        cache_dir: Path | None = None,
        cache_ttl_s: float = 0.0,
        bearer_token: str | None = None,
        transport: httpx.BaseTransport | None = None,
        audit: AuditHook | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.service = service
        self.max_retries = max_retries
        self.cache_dir = cache_dir
        self.cache_ttl_s = cache_ttl_s
        self.bearer_token = bearer_token
        self.audit = audit
        self._sleep = sleep
        self._client = httpx.Client(
            timeout=httpx.Timeout(timeout_s, connect=min(15.0, timeout_s)),
            transport=transport,
            follow_redirects=True,
            headers={"User-Agent": USER_AGENT},
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "NasaHttpClient":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    # ------------------------------------------------------------------ cache
    def _cache_path(self, method: str, url: str, params: Any, body: Any) -> Path | None:
        if self.cache_dir is None or self.cache_ttl_s <= 0:
            return None
        key = json.dumps([method, url, params, body], sort_keys=True, default=str)
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self.cache_dir / self.service / f"{digest}.json"

    def _read_cache(self, path: Path | None) -> Any | None:
        if path is None or not path.is_file():
            return None
        if time.time() - path.stat().st_mtime > self.cache_ttl_s:
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    @staticmethod
    def _write_cache(path: Path | None, payload: Any) -> None:
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(payload), encoding="utf-8")
        temp.replace(path)

    # --------------------------------------------------------------- requests
    def _headers(self, extra: dict[str, str] | None, auth: bool) -> dict[str, str]:
        headers = dict(extra or {})
        if auth and self.bearer_token:
            headers["Authorization"] = f"Bearer {self.bearer_token}"
        return headers

    def _record(self, **entry: Any) -> None:
        if self.audit is not None:
            try:
                self.audit({"service": self.service, **entry})
            except Exception:  # auditing must never break a data request
                logger.exception("NASA audit hook failed")

    def request(
        self,
        method: str,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: Any = None,
        headers: dict[str, str] | None = None,
        auth: bool = False,
        basic_auth: tuple[str, str] | None = None,
        stream: bool = False,
    ) -> httpx.Response:
        check_url_allowed(url)
        last_error: str = "unknown error"
        for attempt in range(self.max_retries + 1):
            started = time.perf_counter()
            try:
                request = self._client.build_request(
                    method,
                    url,
                    params=params,
                    json=json_body,
                    headers=self._headers(headers, auth),
                )
                send_kwargs: dict[str, Any] = {"stream": stream}
                if basic_auth is not None:
                    send_kwargs["auth"] = httpx.BasicAuth(*basic_auth)
                response = self._client.send(request, **send_kwargs)
            except httpx.HTTPError as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                self._record(url=url, ok=False, status_code=None, error=last_error,
                             duration_ms=(time.perf_counter() - started) * 1000)
            else:
                elapsed = (time.perf_counter() - started) * 1000
                if response.status_code < 400:
                    self._record(url=url, ok=True, status_code=response.status_code,
                                 duration_ms=elapsed)
                    return response
                body = response.text[:300] if not stream else ""
                last_error = f"HTTP {response.status_code}: {body}".strip()
                self._record(url=url, ok=False, status_code=response.status_code,
                             error=last_error, duration_ms=elapsed)
                if response.status_code not in RETRY_STATUS:
                    response.close()
                    raise NasaApiError(self.service, last_error, response.status_code)
                response.close()
            if attempt < self.max_retries:
                self._sleep(min(30.0, 2.0**attempt))
        raise NasaApiError(self.service, f"Échec après {self.max_retries + 1} tentatives : {last_error}")

    def get_json(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        auth: bool = False,
        use_cache: bool = True,
    ) -> Any:
        cache_path = self._cache_path("GET", url, params, None) if use_cache else None
        cached = self._read_cache(cache_path)
        if cached is not None:
            self._record(url=url, ok=True, status_code=200, cached=True, duration_ms=0.0)
            return cached
        response = self.request("GET", url, params=params, auth=auth)
        payload = self._decode(response)
        self._write_cache(cache_path, payload)
        return payload

    def post_json(self, url: str, body: Any, *, auth: bool = False,
                  basic_auth: tuple[str, str] | None = None) -> Any:
        response = self.request("POST", url, json_body=body, auth=auth, basic_auth=basic_auth)
        return self._decode(response)

    def _decode(self, response: httpx.Response) -> Any:
        try:
            return response.json()
        except ValueError as exc:
            raise NasaApiError(self.service, "Réponse non JSON reçue.", response.status_code) from exc

    def download(self, url: str, destination: Path, *, max_bytes: int, auth: bool = True) -> Path:
        """Stream a file to ``destination`` atomically, enforcing ``max_bytes``."""
        destination.parent.mkdir(parents=True, exist_ok=True)
        temp = destination.with_name(destination.name + ".part")
        response = self.request("GET", url, auth=auth, stream=True)
        written = 0
        try:
            with temp.open("wb") as handle:
                for chunk in response.iter_bytes(chunk_size=1024 * 1024):
                    written += len(chunk)
                    if written > max_bytes:
                        raise NasaApiError(self.service, "Téléchargement au-delà de la taille autorisée.")
                    handle.write(chunk)
        except BaseException:
            temp.unlink(missing_ok=True)
            raise
        finally:
            response.close()
        temp.replace(destination)
        return destination
