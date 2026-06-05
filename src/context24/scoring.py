from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Iterable, Mapping

import numpy as np

from src.context24.calibration import Calibration
from src.context24.normalizers import score_symbol, signed_strength
from src.context24.quality import quality_check
from src.context24.registry import CATEGORY_CAPS, METRICS, metric_key
from src.context24.schema import FinalSignal, MetricRow, MetricScore


def _metric_reliability(key: str, calibration: Mapping[str, Calibration]) -> tuple[float, str | None]:
    c = calibration.get(key)
    if c is None:
        return 0.0, "missing_historical_calibration"
    if c.reliability <= 0:
        return 0.0, f"insufficient_or_zero_edge:n={c.n_samples}"
    return c.reliability, None


def score_row(row: MetricRow, calibration: Mapping[str, Calibration] | None = None, now: datetime | None = None) -> MetricScore:
    calibration = calibration or {}
    key = metric_key(row.source, row.metric)
    definition = METRICS.get(key)
    quality, reasons = quality_check(row, definition, now=now)
    if definition is None:
        return MetricScore(key, row.source, row.metric, "unknown", row.value, row.delta_24h, quality, 0.0, 0.0, "0", False, reasons)

    cal_rel, cal_reason = _metric_reliability(key, calibration)
    reasons_list = list(reasons)
    if cal_reason:
        reasons_list.append(cal_reason)

    base = signed_strength(row, definition)
    confidence = quality * cal_rel * definition.reliability_weight
    signal = base * confidence
    usable = quality > 0 and confidence > 0 and abs(signal) > 0
    return MetricScore(
        key=key,
        source=row.source,
        metric=row.metric,
        category=definition.category,
        value=row.value,
        delta_24h=row.delta_24h,
        data_quality=quality,
        signal_score=signal,
        confidence=confidence,
        symbol=score_symbol(signal),
        usable=usable,
        reasons=tuple(reasons_list),
    )


def _aggregate_categories(scores: Iterable[MetricScore]) -> tuple[dict[str, float], dict[str, float]]:
    by_cat: dict[str, list[MetricScore]] = defaultdict(list)
    for s in scores:
        if s.usable:
            by_cat[s.category].append(s)
    cat_scores: dict[str, float] = {}
    cat_conf: dict[str, float] = {}
    for cat, rows in by_cat.items():
        weights = np.array([max(r.confidence, 0.0) for r in rows], dtype=float)
        vals = np.array([r.signal_score for r in rows], dtype=float)
        if weights.sum() <= 0:
            continue
        raw = float(np.average(vals, weights=weights))
        cap = CATEGORY_CAPS.get(cat, 0.10)
        cat_scores[cat] = max(-cap, min(cap, raw * cap))
        cat_conf[cat] = float(min(1.0, weights.mean()))
    return cat_scores, cat_conf


def _status(final_score: float, confidence: float, agreeing_categories: int, reasons: list[str]) -> str:
    if reasons:
        return "REJECTED"
    if confidence < 0.20 or agreeing_categories < 3:
        return "WATCH"
    if final_score >= 0.10:
        return "CONFIRMED_LONG"
    if final_score <= -0.10:
        return "CONFIRMED_SHORT"
    return "WATCH"


def score_table(
    rows: Iterable[Mapping[str, object] | MetricRow],
    calibration: Mapping[str, Calibration] | None = None,
    now: datetime | None = None,
) -> FinalSignal:
    metric_rows = [r if isinstance(r, MetricRow) else MetricRow.from_mapping(r) for r in rows]
    scored = tuple(score_row(r, calibration=calibration, now=now) for r in metric_rows)
    cat_scores, cat_conf = _aggregate_categories(scored)
    final_score = float(sum(cat_scores.values())) if cat_scores else 0.0
    confidence = float(np.mean(list(cat_conf.values()))) if cat_conf else 0.0

    direction = "LONG" if final_score > 0 else ("SHORT" if final_score < 0 else "NEUTRAL")
    agreeing = sum(1 for v in cat_scores.values() if (v > 0 and final_score > 0) or (v < 0 and final_score < 0))

    reasons: list[str] = []
    usable_count = sum(1 for s in scored if s.usable)
    if usable_count == 0:
        reasons.append("no_usable_calibrated_rows")
    if len(cat_scores) < 3:
        reasons.append(f"insufficient_independent_categories:{len(cat_scores)}<3")
    if confidence < 0.20:
        reasons.append(f"low_confidence:{confidence:.3f}<0.20")

    return FinalSignal(
        status=_status(final_score, confidence, agreeing, reasons),
        final_score=final_score,
        confidence=confidence,
        direction=direction,
        category_scores=cat_scores,
        category_confidence=cat_conf,
        rows=scored,
        reasons=tuple(reasons),
    )
