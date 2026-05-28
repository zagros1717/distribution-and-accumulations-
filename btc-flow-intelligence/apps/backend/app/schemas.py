"""Pydantic schemas shared across adapters, scoring, and the API layer."""

from __future__ import annotations

import datetime as dt

from pydantic import BaseModel, Field


class SignalReading(BaseModel):
    """A single normalized metric emitted by an adapter."""

    category: str = Field(..., description="Stable category key (see scoring_spec).")
    source: str
    metric: str
    value: float | None = None
    change_24h: float | None = None
    # -2..+2; the adapter's own interpretation of this metric's directionality.
    score: int = 0
    is_live: bool = False
    raw: dict | None = None


class CategoryScore(BaseModel):
    category: str
    label: str
    weight: float
    score: float          # mean of the category's signal scores, -2..+2
    weighted: float       # score * weight
    is_live: bool         # any live signal present
    signal_count: int


class DashboardResponse(BaseModel):
    snapshot_id: int
    timestamp: dt.datetime
    btc_price: float | None
    btc_change_24h: float | None
    final_score: float
    verdict: str
    confidence: str
    data_quality: float
    categories: list[CategoryScore]


class SignalsResponse(BaseModel):
    snapshot_id: int
    timestamp: dt.datetime
    signals: list[SignalReading]


class ReportResponse(BaseModel):
    snapshot_id: int
    created_at: dt.datetime
    markdown: str


class HistoryPoint(BaseModel):
    snapshot_id: int
    timestamp: dt.datetime
    btc_price: float | None
    final_score: float
    verdict: str
    confidence: str


class HealthResponse(BaseModel):
    status: str
    environment: str
    mock_mode: bool
    sources: dict[str, str]   # source -> "live" | "mock"
    last_snapshot: dt.datetime | None
