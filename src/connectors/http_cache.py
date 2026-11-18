"""Polite HTTP GET with retries + a file cache under data/raw/<source>/ (shared by the statistics connectors).

Network happens only in fetch commands; models and loaders read the cache offline. Each cached body has a
sidecar <file>.meta.json with the exact URL and retrieval time (provenance for rule 1).
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

RETRY_STATUS = (429, 500, 502, 503, 504)


class PoliteClient:
    """GET with a minimum interval between requests and exponential backoff on 429/5xx/transport errors."""

    def __init__(self, min_interval: float = 0.5, retries: int = 5, timeout: float = 120,
                 headers: dict | None = None, client: httpx.Client | None = None, sleep=time.sleep):
        self._client = client or httpx.Client(timeout=timeout, headers=headers or {}, follow_redirects=True)
        self._min_interval = min_interval
        self._retries = retries
        self._sleep = sleep
        self._last = 0.0

    def close(self) -> None:
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def get(self, url: str, params: dict | None = None) -> httpx.Response:
        delay = 2.0
        for attempt in range(self._retries):
            wait = self._min_interval - (time.monotonic() - self._last)
            if wait > 0:
                self._sleep(wait)
            self._last = time.monotonic()
            try:
                resp = self._client.get(url, params=params)
                if resp.status_code in RETRY_STATUS:
                    raise httpx.HTTPStatusError(f"HTTP {resp.status_code}", request=resp.request, response=resp)
                resp.raise_for_status()
                return resp
            except (httpx.TransportError, httpx.HTTPStatusError) as e:
                if isinstance(e, httpx.HTTPStatusError) and e.response.status_code not in RETRY_STATUS:
                    raise
                if attempt == self._retries - 1:
                    raise
                retry_after = e.response.headers.get("Retry-After") if isinstance(e, httpx.HTTPStatusError) else None
                self._sleep(float(retry_after) if retry_after and retry_after.isdigit() else delay)
                delay *= 2
        raise AssertionError("unreachable")


def cache_file(cache_dir: Path, key: str, suffix: str) -> Path:
    """Stable file name for a request key (URL + params)."""
    return cache_dir / f"{hashlib.sha256(key.encode()).hexdigest()[:20]}{suffix}"


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def write_cached(path: Path, body: bytes, url: str, **meta) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    Path(str(path) + ".meta.json").write_text(json.dumps({"url": url, "retrieved_at": _now().isoformat(), **meta},
                                                         indent=1))


def read_meta(path: Path) -> dict:
    """{'url', 'retrieved_at' (datetime), ...} for a cached body."""
    m = json.loads(Path(str(path) + ".meta.json").read_text())
    m["retrieved_at"] = datetime.fromisoformat(m["retrieved_at"])
    return m


def cached_get(client: PoliteClient | None, url: str, params, cache_dir: Path, suffix: str,
               refresh: bool = False, secret_params: dict | None = None) -> tuple[bytes, dict]:
    """(body, meta) from cache, else fetched and cached. client=None means offline (raises if not cached).

    secret_params (API keys, contact e-mail) are sent but never part of the cache key or the stored URL.
    """
    key = str(httpx.URL(url, params=params or {}))
    path = cache_file(cache_dir, key, suffix)
    if path.exists() and not refresh:
        return path.read_bytes(), read_meta(path)
    if client is None:
        raise FileNotFoundError(f"not cached and offline: {key}")
    sent = list(params.items() if isinstance(params, dict) else params or []) + list((secret_params or {}).items())
    resp = client.get(url, params=sent)
    write_cached(path, resp.content, key)
    return resp.content, read_meta(path)
