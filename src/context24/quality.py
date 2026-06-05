from __future__ import annotations

from datetime import datetime, timezone

from src.context24.registry import MetricDefinition, metric_key
from src.context24.schema import MetricRow


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def quality_check(row: MetricRow, definition: MetricDefinition | None, now: datetime | None = None) -> tuple[float, tuple[str, ...]]:
    now = now or utcnow()
    reasons: list[str] = []
    quality = 1.0

    if definition is None:
        return 0.0, (f"unknown_metric:{metric_key(row.source, row.metric)}",)

    if row.value is None:
        reasons.append("missing_value")
        quality = 0.0

    if row.as_of is None:
        reasons.append("missing_as_of")
        quality *= 0.0
    else:
        lag_min = max((now - row.as_of).total_seconds() / 60.0, 0.0)
        if lag_min > definition.freshness_minutes:
            reasons.append(f"stale:{lag_min:.1f}m>{definition.freshness_minutes}m")
            quality *= max(0.0, 1.0 - (lag_min - definition.freshness_minutes) / max(definition.freshness_minutes, 1))

    if row.fetched_at is None:
        reasons.append("missing_fetched_at")
        quality *= 0.75
    elif row.as_of is not None and row.fetched_at < row.as_of:
        reasons.append("fetched_before_as_of")
        quality *= 0.5

    if definition.requires_delta and row.delta_24h is None:
        reasons.append("missing_required_delta_24h")
        quality = 0.0

    if row.unit and definition.unit and row.unit.lower() != definition.unit.lower():
        reasons.append(f"unit_mismatch:{row.unit}!={definition.unit}")
        quality *= 0.5

    return max(0.0, min(1.0, quality)), tuple(reasons)
