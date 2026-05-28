"""Time helpers. Everything in this project is UTC."""
from __future__ import annotations

from datetime import datetime, timezone, timedelta


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def parse_iso_utc(s: str) -> datetime:
    """Parse an ISO-8601 timestamp and force UTC tzinfo."""
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def from_unix_seconds(sec: float) -> datetime:
    return datetime.fromtimestamp(sec, tz=timezone.utc)


def from_unix_millis(ms: float) -> datetime:
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)


def floor_to_ms_bucket(dt: datetime, bucket_ms: int) -> datetime:
    """
    Floor `dt` down to the nearest `bucket_ms` boundary.
    Used by the feature engine to align rows on regular intervals.
    """
    epoch_ms = int(dt.timestamp() * 1000)
    floored = (epoch_ms // bucket_ms) * bucket_ms
    return from_unix_millis(floored)


def daterange(start: datetime, end: datetime):
    """Yield each UTC date between two datetimes (inclusive)."""
    d = start.date()
    last = end.date()
    while d <= last:
        yield d
        d = d + timedelta(days=1)
