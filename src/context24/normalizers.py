from __future__ import annotations

import math

from src.context24.registry import MetricDefinition
from src.context24.schema import MetricRow


def signed_strength(row: MetricRow, definition: MetricDefinition) -> float:
    """Return heuristic signed strength in [-1, 1].

    Historical calibration later scales this down unless the metric has proven
    edge. This function only encodes the immediate direction of the observation.
    """
    value = row.value
    delta = row.delta_24h
    if value is None:
        return 0.0

    raw = 0.0
    if definition.requires_delta:
        raw = 0.0 if delta is None else delta
    elif definition.direction_rule in ("positive_is_bullish", "negative_is_bullish"):
        raw = delta if delta is not None else value
    elif definition.direction_rule in ("high_is_bearish", "low_is_bullish"):
        raw = value
    else:
        return 0.0

    # Compress arbitrary units without hand-tuned thresholds.
    mag = math.tanh(abs(float(raw)))
    sign = 1.0 if raw > 0 else (-1.0 if raw < 0 else 0.0)

    if definition.direction_rule in ("negative_is_bullish", "high_is_bearish"):
        sign *= -1.0
    if definition.direction_rule == "low_is_bullish":
        # Low value bullish, high value bearish. Use distance from a neutral
        # midpoint for bounded sentiment-style indexes when possible.
        if 0 <= value <= 100:
            sign = 1.0 if value < 50 else (-1.0 if value > 50 else 0.0)
            mag = min(abs(value - 50.0) / 50.0, 1.0)
        else:
            sign *= -1.0

    return max(-1.0, min(1.0, sign * mag))


def score_symbol(score: float) -> str:
    if score >= 0.66:
        return "++"
    if score >= 0.25:
        return "+"
    if score <= -0.66:
        return "--"
    if score <= -0.25:
        return "-"
    return "0"
