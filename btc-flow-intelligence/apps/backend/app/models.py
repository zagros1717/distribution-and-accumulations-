"""ORM models: snapshots, signals, reports."""

from __future__ import annotations

import datetime as dt
from typing import Any

from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base


def _utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


class Snapshot(Base):
    __tablename__ = "snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, server_default=func.now(), index=True
    )
    btc_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    btc_change_24h: Mapped[float | None] = mapped_column(Float, nullable=True)
    final_score: Mapped[float] = mapped_column(Float, nullable=False)
    verdict: Mapped[str] = mapped_column(String(32), nullable=False)
    confidence: Mapped[str] = mapped_column(String(16), nullable=False)
    # 0..1 — fraction of weighted categories backed by live (non-mock) data.
    data_quality: Mapped[float] = mapped_column(Float, default=0.0)

    signals: Mapped[list["Signal"]] = relationship(
        back_populates="snapshot", cascade="all, delete-orphan"
    )
    report: Mapped["Report | None"] = relationship(
        back_populates="snapshot", cascade="all, delete-orphan", uselist=False
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "btc_price": self.btc_price,
            "btc_change_24h": self.btc_change_24h,
            "final_score": round(self.final_score, 4),
            "verdict": self.verdict,
            "confidence": self.confidence,
            "data_quality": round(self.data_quality, 4),
        }


class Signal(Base):
    __tablename__ = "signals"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    snapshot_id: Mapped[int] = mapped_column(
        ForeignKey("snapshots.id", ondelete="CASCADE"), index=True
    )
    category: Mapped[str] = mapped_column(String(48), nullable=False, index=True)
    source: Mapped[str] = mapped_column(String(48), nullable=False)
    metric: Mapped[str] = mapped_column(String(96), nullable=False)
    value: Mapped[float | None] = mapped_column(Float, nullable=True)
    change_24h: Mapped[float | None] = mapped_column(Float, nullable=True)
    score: Mapped[int] = mapped_column(Integer, nullable=False)  # -2..+2
    is_live: Mapped[bool] = mapped_column(default=False)
    raw_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    snapshot: Mapped["Snapshot"] = relationship(back_populates="signals")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "snapshot_id": self.snapshot_id,
            "category": self.category,
            "source": self.source,
            "metric": self.metric,
            "value": self.value,
            "change_24h": self.change_24h,
            "score": self.score,
            "is_live": self.is_live,
            "raw_json": self.raw_json,
        }


class Report(Base):
    __tablename__ = "reports"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    snapshot_id: Mapped[int] = mapped_column(
        ForeignKey("snapshots.id", ondelete="CASCADE"), unique=True, index=True
    )
    markdown_report: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, server_default=func.now()
    )

    snapshot: Mapped["Snapshot"] = relationship(back_populates="report")
