"""
Base class for all data-source adapters.

Design goals (from the brief):
  * fetch data            -> `_fetch_live`
  * normalize output      -> adapters return list[SignalReading]
  * structured schema     -> SignalReading
  * handle errors         -> never raises to the pipeline; degrades to mock
  * support retries       -> `_request` with exponential backoff
  * support missing data  -> per-signal `score=0`, value=None tolerated

An adapter is "live" when it has the credentials it needs AND mock_mode is off
AND a live fetch succeeds. Otherwise it serves mock data and is flagged so the
confidence/data-quality engine can discount it.
"""

from __future__ import annotations

import abc
import asyncio
import logging
import random

import httpx

from app.config import get_settings
from app.schemas import SignalReading

logger = logging.getLogger(__name__)
settings = get_settings()


class SourceAdapter(abc.ABC):
    name: str = "base"
    category: str = "uncategorized"
    timeout: float = 10.0
    max_retries: int = 3

    #: Names of settings attributes that must be truthy for live mode.
    required_keys: tuple[str, ...] = ()

    def __init__(self) -> None:
        # A per-adapter RNG seeded by name keeps mock data stable-ish per source
        # while still varying run-to-run, so the dashboard looks alive.
        self._rng = random.Random()

    # ------------------------------------------------------------------ public

    @property
    def has_credentials(self) -> bool:
        return all(getattr(settings, k, None) for k in self.required_keys)

    @property
    def can_go_live(self) -> bool:
        return (not settings.mock_mode) and self.has_credentials

    async def collect(self) -> list[SignalReading]:
        """Return normalized readings, never raising."""
        if not self.can_go_live:
            reason = "mock_mode" if settings.mock_mode else "missing_credentials"
            logger.debug("[%s] serving mock (%s)", self.name, reason)
            return self._stamp(self._mock(), live=False)
        try:
            readings = await self._fetch_live()
            if not readings:
                logger.warning("[%s] live fetch returned no data; using mock", self.name)
                return self._stamp(self._mock(), live=False)
            return self._stamp(readings, live=True)
        except Exception as exc:  # graceful degradation — one source can't crash the run
            logger.warning("[%s] live fetch failed (%s); using mock", self.name, exc)
            return self._stamp(self._mock(), live=False)

    # --------------------------------------------------------------- subclass API

    @abc.abstractmethod
    async def _fetch_live(self) -> list[SignalReading]:
        """Hit the real API and return normalized readings. May raise."""

    @abc.abstractmethod
    def _mock(self) -> list[SignalReading]:
        """Return realistic mock readings. Must not raise."""

    # ------------------------------------------------------------------ helpers

    def _stamp(self, readings: list[SignalReading], *, live: bool) -> list[SignalReading]:
        for r in readings:
            r.is_live = live
            if not r.category:
                r.category = self.category
            if not r.source:
                r.source = self.name
        return readings

    async def _request(self, method: str, url: str, **kwargs) -> httpx.Response:
        """HTTP request with exponential backoff + jitter on 5xx/timeouts."""
        last_exc: Exception | None = None
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for attempt in range(1, self.max_retries + 1):
                try:
                    resp = await client.request(method, url, **kwargs)
                    if resp.status_code >= 500:
                        raise httpx.HTTPStatusError(
                            f"server error {resp.status_code}", request=resp.request, response=resp
                        )
                    resp.raise_for_status()
                    return resp
                except (httpx.TimeoutException, httpx.HTTPStatusError, httpx.TransportError) as exc:
                    last_exc = exc
                    if attempt == self.max_retries:
                        break
                    backoff = (2 ** (attempt - 1)) + random.random()
                    logger.debug("[%s] retry %d/%d in %.1fs", self.name, attempt, self.max_retries, backoff)
                    await asyncio.sleep(backoff)
        assert last_exc is not None
        raise last_exc

    # mock-data conveniences -------------------------------------------------

    def _r(self, lo: float, hi: float, nd: int = 2) -> float:
        return round(self._rng.uniform(lo, hi), nd)

    def _score_from_change(self, change: float, mild: float, strong: float) -> int:
        """Map a % change to a -2..+2 score using two thresholds."""
        if change >= strong:
            return 2
        if change >= mild:
            return 1
        if change <= -strong:
            return -2
        if change <= -mild:
            return -1
        return 0
