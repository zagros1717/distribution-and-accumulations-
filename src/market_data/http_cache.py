"""Tiny on-disk HTTP JSON cache used to avoid repeated paid/vendor calls.

The cache is deliberately simple and transparent: every response is stored as a
JSON envelope under data/http_cache. It is safe for research jobs, cron jobs, and
local notebooks, and it keeps repeated dashboard refreshes from hammering APIs.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Optional
from urllib.parse import urlencode

import aiohttp


class HTTPCache:
    def __init__(self, cache_dir: str | Path, default_ttl_seconds: int = 300, enabled: bool = True) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.default_ttl_seconds = int(default_ttl_seconds)
        self.enabled = bool(enabled)

    @staticmethod
    def _now() -> datetime:
        return datetime.now(timezone.utc)

    @staticmethod
    def _canonical_params(params: Optional[Mapping[str, Any]]) -> str:
        if not params:
            return ""
        clean = {k: v for k, v in params.items() if v is not None}
        return urlencode(sorted(clean.items()), doseq=True)

    def _path_for(self, url: str, params: Optional[Mapping[str, Any]]) -> Path:
        key = f"{url}?{self._canonical_params(params)}"
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.json"

    def _read_fresh(self, path: Path) -> Optional[Any]:
        if not self.enabled or not path.exists():
            return None
        try:
            envelope = json.loads(path.read_text())
            expires_at = datetime.fromisoformat(envelope["expires_at"])
            if expires_at.tzinfo is None:
                expires_at = expires_at.replace(tzinfo=timezone.utc)
            if expires_at > self._now():
                return envelope["payload"]
        except Exception:
            return None
        return None

    def _write(self, path: Path, url: str, params: Optional[Mapping[str, Any]], payload: Any, ttl_seconds: int) -> None:
        if not self.enabled:
            return
        now = self._now()
        envelope = {
            "url": url,
            "params": {k: v for k, v in (params or {}).items() if v is not None},
            "fetched_at": now.isoformat(),
            "expires_at": (now + timedelta(seconds=int(ttl_seconds))).isoformat(),
            "payload": payload,
        }
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(envelope, sort_keys=True, default=str))
        tmp.replace(path)

    async def get_json(
        self,
        session: aiohttp.ClientSession,
        url: str,
        *,
        params: Optional[Mapping[str, Any]] = None,
        headers: Optional[Mapping[str, str]] = None,
        ttl_seconds: Optional[int] = None,
    ) -> Any:
        ttl = int(ttl_seconds if ttl_seconds is not None else self.default_ttl_seconds)
        path = self._path_for(url, params)
        cached = self._read_fresh(path)
        if cached is not None:
            return cached

        async with session.get(url, params=params, headers=headers, timeout=30) as resp:
            resp.raise_for_status()
            payload = await resp.json(content_type=None)
        self._write(path, url, params, payload, ttl)
        return payload
