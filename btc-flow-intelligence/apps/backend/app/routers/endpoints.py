"""All API endpoints."""

from __future__ import annotations

import datetime as dt

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db import get_db
from app.models import Report, Signal, Snapshot
from app.pipeline import run_pipeline
from app.schemas import (
    CategoryScore,
    DashboardResponse,
    HealthResponse,
    HistoryPoint,
    ReportResponse,
    SignalReading,
    SignalsResponse,
)
from app.sources import all_adapters

router = APIRouter(prefix="/api")
settings = get_settings()


def _latest_snapshot(db: Session) -> Snapshot:
    snap = db.execute(
        select(Snapshot).order_by(Snapshot.timestamp.desc()).limit(1)
    ).scalar_one_or_none()
    if snap is None:
        raise HTTPException(status_code=404, detail="No snapshot yet. POST /api/refresh first.")
    return snap


def _categories_from_signals(signals: list[Signal]) -> list[CategoryScore]:
    # Recompute category view from stored signals so the API stays consistent
    # with the scoring engine without persisting a denormalised copy.
    from app.scoring import _category_scores  # local import to avoid cycle

    readings = [
        SignalReading(
            category=s.category, source=s.source, metric=s.metric, value=s.value,
            change_24h=s.change_24h, score=s.score, is_live=s.is_live, raw=s.raw_json,
        )
        for s in signals
    ]
    return _category_scores(readings)


@router.get("/dashboard", response_model=DashboardResponse)
def dashboard(db: Session = Depends(get_db)) -> DashboardResponse:
    snap = _latest_snapshot(db)
    categories = _categories_from_signals(snap.signals)
    return DashboardResponse(
        snapshot_id=snap.id, timestamp=snap.timestamp,
        btc_price=snap.btc_price, btc_change_24h=snap.btc_change_24h,
        final_score=snap.final_score, verdict=snap.verdict,
        confidence=snap.confidence, data_quality=snap.data_quality,
        categories=categories,
    )


@router.get("/signals", response_model=SignalsResponse)
def signals(db: Session = Depends(get_db)) -> SignalsResponse:
    snap = _latest_snapshot(db)
    readings = [
        SignalReading(
            category=s.category, source=s.source, metric=s.metric, value=s.value,
            change_24h=s.change_24h, score=s.score, is_live=s.is_live, raw=s.raw_json,
        )
        for s in snap.signals
    ]
    return SignalsResponse(snapshot_id=snap.id, timestamp=snap.timestamp, signals=readings)


@router.get("/report/latest", response_model=ReportResponse)
def latest_report(db: Session = Depends(get_db)) -> ReportResponse:
    snap = _latest_snapshot(db)
    report = db.execute(
        select(Report).where(Report.snapshot_id == snap.id)
    ).scalar_one_or_none()
    if report is None:
        raise HTTPException(status_code=404, detail="No report for latest snapshot.")
    return ReportResponse(
        snapshot_id=snap.id, created_at=report.created_at, markdown=report.markdown_report
    )


@router.get("/history", response_model=list[HistoryPoint])
def history(
    limit: int = Query(default=168, ge=1, le=1000),
    db: Session = Depends(get_db),
) -> list[HistoryPoint]:
    rows = db.execute(
        select(Snapshot).order_by(Snapshot.timestamp.desc()).limit(limit)
    ).scalars().all()
    rows = list(reversed(rows))  # chronological for charting
    return [
        HistoryPoint(
            snapshot_id=r.id, timestamp=r.timestamp, btc_price=r.btc_price,
            final_score=r.final_score, verdict=r.verdict, confidence=r.confidence,
        )
        for r in rows
    ]


@router.post("/refresh", response_model=DashboardResponse)
async def refresh(db: Session = Depends(get_db)) -> DashboardResponse:
    snap = await run_pipeline(db)
    categories = _categories_from_signals(snap.signals)
    return DashboardResponse(
        snapshot_id=snap.id, timestamp=snap.timestamp,
        btc_price=snap.btc_price, btc_change_24h=snap.btc_change_24h,
        final_score=snap.final_score, verdict=snap.verdict,
        confidence=snap.confidence, data_quality=snap.data_quality,
        categories=categories,
    )


@router.get("/health", response_model=HealthResponse)
def health(db: Session = Depends(get_db)) -> HealthResponse:
    sources = {
        a.name: ("live" if a.can_go_live else "mock") for a in all_adapters()
    }
    last = db.execute(
        select(Snapshot.timestamp).order_by(Snapshot.timestamp.desc()).limit(1)
    ).scalar_one_or_none()
    return HealthResponse(
        status="ok", environment=settings.environment, mock_mode=settings.mock_mode,
        sources=sources, last_snapshot=last,
    )
