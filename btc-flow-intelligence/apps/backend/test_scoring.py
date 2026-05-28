"""Smoke + unit tests. Run: `pytest` from apps/backend."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

# Use an in-memory-ish SQLite file and force mock mode before importing the app.
os.environ.setdefault("DATABASE_URL", "sqlite:///./test_btcflow.db")
os.environ.setdefault("MOCK_MODE", "true")
os.environ.setdefault("RUN_ON_STARTUP", "false")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.db import SessionLocal, init_db  # noqa: E402
from app.pipeline import gather_signals, run_pipeline  # noqa: E402
from app.scoring import score_signals  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "packages" / "shared"))
from scoring_spec import CATEGORIES, assert_weights_valid, classify  # noqa: E402


def test_weights_sum_to_one():
    assert_weights_valid()
    assert round(sum(c.weight for c in CATEGORIES), 6) == 1.0


def test_thresholds():
    assert classify(0.8).value == "Accumulation"
    assert classify(-0.8).value == "Distribution"
    assert classify(0.0).value == "Mixed/Neutral"


def test_mock_pipeline_end_to_end():
    init_db()
    signals = asyncio.run(gather_signals())
    assert len(signals) > 10  # all adapters contributed

    result = score_signals(signals)
    assert -2.0 <= result.final_score <= 2.0
    assert result.verdict in {"Accumulation", "Distribution", "Mixed/Neutral"}
    # Pure mock mode → nothing verified live → confidence must be Low.
    assert result.data_quality == 0.0
    assert result.confidence == "Low"

    db = SessionLocal()
    try:
        snap = asyncio.run(run_pipeline(db))
        assert snap.id is not None
        assert snap.report is not None
        assert "Final Verdict" in snap.report.markdown_report
    finally:
        db.close()


if __name__ == "__main__":
    test_weights_sum_to_one()
    test_thresholds()
    test_mock_pipeline_end_to_end()
    print("All tests passed ✓")
