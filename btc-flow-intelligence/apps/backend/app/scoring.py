"""
The scoring engine.

Pipeline of responsibility:
  signals (per-metric, -2..+2)
    -> category score (mean of that category's signals, -2..+2)
    -> weighted category score (category score * category weight)
    -> final weighted score (sum of weighted category scores, -2..+2)
    -> verdict (threshold classification)
    -> confidence (data quality x signal agreement)

When mock mode is disabled, non-live fallback signals remain visible for
transparency but are excluded from scores and verdicts.
"""

from __future__ import annotations

import statistics
import os
import sys
from dataclasses import dataclass
from pathlib import Path

# Locate packages/shared robustly across dev and container layouts.
_CANDIDATES = [
    os.environ.get("SHARED_DIR"),
    Path(__file__).resolve().parents[3] / "packages" / "shared",
    Path("/app/packages/shared"),
    Path(__file__).resolve().parent / "_shared",  # vendored fallback
]
for _c in _CANDIDATES:
    if _c and Path(_c).exists():
        if str(_c) not in sys.path:
            sys.path.insert(0, str(_c))
        break

from scoring_spec import (  # type: ignore  # noqa: E402
    CATEGORY_BY_KEY,
    CATEGORIES,
    Confidence,
    classify,
)

from app.config import get_settings  # noqa: E402
from app.schemas import CategoryScore, SignalReading  # noqa: E402

settings = get_settings()


@dataclass
class ScoringResult:
    final_score: float
    verdict: str
    confidence: str
    data_quality: float
    categories: list[CategoryScore]


_CONTEXT_METRICS = {"btc_price", "btc_daily_close", "options_open_interest", "spot_volume", "futures_volume"}


def _category_scores(signals: list[SignalReading]) -> list[CategoryScore]:
    by_cat: dict[str, list[SignalReading]] = {c.key: [] for c in CATEGORIES}
    for s in signals:
        if s.category in by_cat:
            by_cat[s.category].append(s)

    results: list[CategoryScore] = []
    for cat in CATEGORIES:
        readings = by_cat[cat.key]
        eligible = readings if settings.mock_mode else [r for r in readings if r.is_live]
        # Only signals that actually express a view (non-context) count toward
        # the mean. Price/context readings should not drag categories to zero.
        voting = [r for r in eligible if not (r.score == 0 and r.metric in _CONTEXT_METRICS)]
        score = statistics.fmean([r.score for r in voting]) if voting else 0.0
        results.append(
            CategoryScore(
                category=cat.key,
                label=cat.label,
                weight=cat.weight,
                score=round(score, 3),
                weighted=round(score * cat.weight, 4),
                is_live=any(r.is_live for r in readings),
                signal_count=len(readings),
            )
        )
    return results


def _confidence(data_quality: float, categories: list[CategoryScore]) -> Confidence:
    # Agreement: do the categories broadly point the same way?
    directions = [1 if c.score > 0.25 else -1 if c.score < -0.25 else 0 for c in categories if c.signal_count]
    if directions:
        non_zero = [d for d in directions if d != 0]
        agreement = (abs(sum(non_zero)) / len(non_zero)) if non_zero else 0.0
    else:
        agreement = 0.0

    composite = 0.6 * data_quality + 0.4 * agreement
    if data_quality == 0.0:
        # Nothing verified live -> never claim more than LOW confidence.
        return Confidence.LOW
    if composite >= 0.66:
        return Confidence.HIGH
    if composite >= 0.4:
        return Confidence.MEDIUM
    return Confidence.LOW


def score_signals(signals: list[SignalReading]) -> ScoringResult:
    categories = _category_scores(signals)
    final = round(sum(c.weighted for c in categories), 4)

    # Data quality: weight of categories backed by live data divided by total weight present.
    present = [c for c in categories if c.signal_count]
    live_weight = sum(c.weight for c in present if c.is_live)
    total_weight = sum(c.weight for c in present) or 1.0
    data_quality = round(live_weight / total_weight, 4)

    verdict = classify(final)
    confidence = _confidence(data_quality, categories)

    return ScoringResult(
        final_score=final,
        verdict=verdict.value,
        confidence=confidence.value,
        data_quality=data_quality,
        categories=categories,
    )
