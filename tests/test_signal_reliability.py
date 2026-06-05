from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest.simulator import BacktestConfig, run_backtest
from src.models.train_xgboost import _feature_columns, NON_FEATURE_COLS


def test_trainer_excludes_absolute_price_level_features():
    df = pd.DataFrame({
        "ts": pd.date_range("2026-01-01", periods=3, tz="UTC"),
        "exchange": ["x"] * 3,
        "symbol": ["BTCUSD"] * 3,
        "mid_price": [100, 101, 102],
        "best_bid": [99, 100, 101],
        "best_ask": [101, 102, 103],
        "last_trade_price": [100, 101, 102],
        "vwap": [100, 101, 102],
        "microprice": [100.1, 101.1, 102.1],
        "relative_spread": [0.001, 0.001, 0.001],
        "order_book_imbalance_1": [0.55, 0.50, 0.45],
        "label_class": [1, 2, 0],
    })

    cols = _feature_columns(df)

    for price_col in ["mid_price", "best_bid", "best_ask", "last_trade_price", "vwap", "microprice"]:
        assert price_col in NON_FEATURE_COLS
        assert price_col not in cols
    assert "relative_spread" in cols
    assert "order_book_imbalance_1" in cols


def test_backtest_skips_trade_when_entry_snapshot_is_stale():
    df = pd.DataFrame({
        "ts": pd.to_datetime(["2026-01-01T00:00:00Z", "2026-01-01T00:00:10Z", "2026-01-01T00:00:15Z"], utc=True),
        "mid_price": [100.0, 110.0, 111.0],
        "spread": [0.1, 0.1, 0.1],
        "is_valid": [True, True, True],
    })
    preds = np.array([1, 0, 0])
    conf = np.array([0.99, 0.0, 0.0])

    result = run_backtest(
        df, preds, conf, horizon_s=5,
        config=BacktestConfig(latency_ms=250, min_confidence=0.5, max_snapshot_delay_ms=500),
    )

    assert result.summary["n_trades"] == 0
    assert result.summary["skipped_stale_entry"] == 1


def test_backtest_skips_trade_when_exit_snapshot_is_stale():
    df = pd.DataFrame({
        "ts": pd.to_datetime(["2026-01-01T00:00:00Z", "2026-01-01T00:00:00.250Z", "2026-01-01T00:00:20Z"], utc=True),
        "mid_price": [100.0, 100.0, 120.0],
        "spread": [0.1, 0.1, 0.1],
        "is_valid": [True, True, True],
    })
    preds = np.array([1, 0, 0])
    conf = np.array([0.99, 0.0, 0.0])

    result = run_backtest(
        df, preds, conf, horizon_s=5,
        config=BacktestConfig(latency_ms=250, min_confidence=0.5, max_snapshot_delay_ms=500),
    )

    assert result.summary["n_trades"] == 0
    assert result.summary["skipped_stale_exit"] == 1
