"""Canonical records for cheap public market/context metrics."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Optional


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def from_unix_millis(ms: int | str | float | None, *, fallback: Optional[datetime] = None) -> datetime:
    """Parse an exchange millisecond timestamp into an aware UTC datetime."""
    if ms in (None, ""):
        return fallback or utcnow()
    return datetime.fromtimestamp(float(ms) / 1000.0, tz=timezone.utc)


def parse_float(value: Any) -> Optional[float]:
    """Best-effort float parser that preserves missing values as None."""
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class MarketMetricRecord:
    """One normalized market/context metric observation.

    `value` is the primary numeric value. Ratios or related fields that do not
    fit into a single scalar stay in `raw_payload`, so no source detail is lost.
    """

    source: str
    metric: str
    symbol: str
    ts: datetime
    value: Optional[float]
    unit: str = ""
    raw_payload: Mapping[str, Any] = field(default_factory=dict)
    fetched_at: datetime = field(default_factory=utcnow)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["ts"] = self.ts.astimezone(timezone.utc).isoformat()
        d["fetched_at"] = self.fetched_at.astimezone(timezone.utc).isoformat()
        d["raw_payload"] = json.dumps(dict(self.raw_payload), sort_keys=True, default=str)
        return d
