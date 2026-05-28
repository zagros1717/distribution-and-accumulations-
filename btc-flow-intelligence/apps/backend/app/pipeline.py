"""
End-to-end refresh pipeline.

  1. fetch all sources concurrently (each degrades to mock on failure)
  2. normalize -> list[SignalReading]
  3. score categories + final verdict + confidence
  4. persist snapshot + signals
  5. generate + store markdown report
  6. fire optional alerts on verdict change
"""

from __future__ import annotations

import asyncio
import logging

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.alerts import maybe_alert_verdict_change
from app.models import Report, Signal, Snapshot
from app.report import generate_report
from app.schemas import SignalReading
from app.scoring import score_signals
from app.sources import all_adapters

logger = logging.getLogger(__name__)


async def gather_signals() -> list[SignalReading]:
    adapters = all_adapters()
    results = await asyncio.gather(*(a.collect() for a in adapters), return_exceptions=True)
    signals: list[SignalReading] = []
    for adapter, res in zip(adapters, results):
        if isinstance(res, Exception):
            logger.error("[%s] collect() raised unexpectedly: %s", adapter.name, res)
            continue
        signals.extend(res)
    return signals


def _extract_price(signals: list[SignalReading]) -> tuple[float | None, float | None]:
    for s in signals:
        if s.metric == "btc_price":
            return s.value, s.change_24h
    return None, None


async def run_pipeline(db: Session) -> Snapshot:
    signals = await gather_signals()
    btc_price, btc_change = _extract_price(signals)
    result = score_signals(signals)

    snapshot = Snapshot(
        btc_price=btc_price,
        btc_change_24h=btc_change,
        final_score=result.final_score,
        verdict=result.verdict,
        confidence=result.confidence,
        data_quality=result.data_quality,
    )
    db.add(snapshot)
    db.flush()  # assign snapshot.id

    for s in signals:
        db.add(Signal(
            snapshot_id=snapshot.id, category=s.category, source=s.source,
            metric=s.metric, value=s.value, change_24h=s.change_24h,
            score=s.score, is_live=s.is_live, raw_json=s.raw,
        ))

    # 7-day context for the report
    history_rows = db.execute(
        select(Snapshot).order_by(Snapshot.timestamp.desc()).limit(8)
    ).scalars().all()
    history = [r.to_dict() for r in history_rows if r.id != snapshot.id][:7]

    markdown = generate_report(
        result=result, signals=signals,
        btc_price=btc_price, btc_change_24h=btc_change, history=history,
    )
    db.add(Report(snapshot_id=snapshot.id, markdown_report=markdown))
    db.commit()
    db.refresh(snapshot)

    previous_verdict = history[0]["verdict"] if history else None
    await maybe_alert_verdict_change(
        previous_verdict, result.verdict, result.final_score, result.confidence
    )

    logger.info(
        "Snapshot %s: %s (%.2f, %s, dq=%.2f)",
        snapshot.id, result.verdict, result.final_score, result.confidence, result.data_quality,
    )
    return snapshot
