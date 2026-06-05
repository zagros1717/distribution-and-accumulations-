from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Optional


def parse_ts(value: Any) -> Optional[datetime]:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        text = str(value)
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(text)
        except ValueError:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def parse_float(value: Any) -> Optional[float]:
    if value in (None, "", "—", "-"):
        return None
    try:
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class MetricRow:
    source: str
    metric: str
    value: Optional[float]
    delta_24h: Optional[float]
    as_of: Optional[datetime]
    fetched_at: Optional[datetime]
    unit: str = ""
    raw: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> "MetricRow":
        source = str(row.get("source", row.get("Source", ""))).strip().lower()
        metric = str(row.get("metric", row.get("Metric", ""))).strip().lower()
        return cls(
            source=source,
            metric=metric,
            value=parse_float(row.get("value", row.get("Value"))),
            delta_24h=parse_float(row.get("delta_24h", row.get("Δ24h", row.get("delta")))),
            as_of=parse_ts(row.get("as_of", row.get("as_of_timestamp", row.get("timestamp")))),
            fetched_at=parse_ts(row.get("fetched_at", row.get("fetched_at_timestamp"))),
            unit=str(row.get("unit", row.get("Unit", ""))).strip(),
            raw=dict(row),
        )


@dataclass(frozen=True)
class MetricScore:
    key: str
    source: str
    metric: str
    category: str
    value: Optional[float]
    delta_24h: Optional[float]
    data_quality: float
    signal_score: float
    confidence: float
    symbol: str
    usable: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class FinalSignal:
    status: str
    final_score: float
    confidence: float
    direction: str
    category_scores: Mapping[str, float]
    category_confidence: Mapping[str, float]
    rows: tuple[MetricScore, ...]
    reasons: tuple[str, ...]
