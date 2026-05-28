"""
interval_ms label isolation tests.

Before this fix, `labels_path()` didn't include interval_ms in the output
directory. That meant labels generated from 100ms features and labels
generated from 1000ms features were written to the same parquet directory
and silently mixed at training time — the model trained on a frankenset of
incompatible rows.

We verify:

  1. labels_path('foo', ..., interval_ms=100, ...) and the same call with
     interval_ms=1000 produce DIFFERENT directories.
  2. Generating labels with interval_ms=100 then interval_ms=1000 leaves two
     distinct parquet trees; reading each tree back gives only its own rows
     (no contamination).
  3. The interval_ms is recorded inside each label row (so even if someone
     reads two trees and concatenates, the column can be used to split).
  4. The round-trip threshold from compute_threshold_bps is twice the one-way
     sum — i.e. it matches the backtester's cost model.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.labels.label_engine import (
    LABEL_SCHEMA, compute_threshold_bps, generate_labels,
)
from src.storage.parquet_store import (
    labels_path, features_path, ParquetWriter,
)


# ---------------------------------------------------------------------------
# Path isolation
# ---------------------------------------------------------------------------

def test_labels_path_includes_interval_ms(tmp_path):
    p100 = labels_path(tmp_path, "bitfinex", "BTCUSD",
                       datetime(2026, 5, 22, tzinfo=timezone.utc),
                       interval_ms=100, horizon_s=5)
    p1000 = labels_path(tmp_path, "bitfinex", "BTCUSD",
                        datetime(2026, 5, 22, tzinfo=timezone.utc),
                        interval_ms=1000, horizon_s=5)
    assert p100 != p1000, "labels for different intervals must live in different dirs"
    assert "interval_ms=100" in str(p100)
    assert "interval_ms=1000" in str(p1000)
    # Horizon partitioning is preserved.
    assert "horizon_s=5" in str(p100)
    assert "horizon_s=5" in str(p1000)


def test_labels_path_distinct_for_different_horizons_within_interval(tmp_path):
    d = datetime(2026, 5, 22, tzinfo=timezone.utc)
    p1 = labels_path(tmp_path, "bitfinex", "BTCUSD", d, interval_ms=1000, horizon_s=1)
    p5 = labels_path(tmp_path, "bitfinex", "BTCUSD", d, interval_ms=1000, horizon_s=5)
    assert p1 != p5


# ---------------------------------------------------------------------------
# Round-trip threshold consistency with backtester
# ---------------------------------------------------------------------------

def test_threshold_default_is_round_trip():
    """Default must be 2x the one-way cost so labels match backtest reality."""
    cost = {"taker_fee": 10, "half_spread_buffer": 3, "slippage_buffer": 2}
    one_way = compute_threshold_bps(cost, round_trip=False)
    two_way = compute_threshold_bps(cost)
    assert one_way == 15.0
    assert two_way == 30.0
    assert compute_threshold_bps(cost) == two_way, "default must be round-trip"


def test_threshold_matches_backtest_round_trip_costs():
    """
    The label threshold must equal what the backtester actually pays on a
    round trip: 2 * (taker_fee + half_spread + slippage). Pin this down so a
    future refactor doesn't drift the two values apart.
    """
    cost = {"taker_fee": 10, "half_spread_buffer": 3, "slippage_buffer": 2}
    # Backtest cost model (entry + exit):
    backtest_round_trip_bps = 2 * (
        cost["taker_fee"] + cost["half_spread_buffer"] + cost["slippage_buffer"]
    )
    assert compute_threshold_bps(cost) == backtest_round_trip_bps


# ---------------------------------------------------------------------------
# End-to-end: generate labels at two intervals, verify isolation
# ---------------------------------------------------------------------------

def _write_toy_features(tmp_path: Path, exchange: str, symbol: str,
                         dt: datetime, interval_ms: int, n_rows: int,
                         seed: int) -> None:
    """Write a minimal features parquet file the label engine can read."""
    rng = np.random.default_rng(seed)
    # Spacing matches interval_ms so the label engine's merge_asof finds
    # future rows within tolerance.
    start = pd.Timestamp(dt).normalize() + pd.Timedelta(hours=12)
    # Ensure UTC awareness if tz_convert was lost during normalize
    if start.tzinfo is None:
        start = start.tz_localize("UTC")
    ts = pd.date_range(start, periods=n_rows, freq=f"{interval_ms}ms", tz="UTC")
    # Drift the price slightly each step so some moves clear the round-trip
    # threshold (30 bps on 50_000 ≈ 150) and others don't.
    increments = rng.normal(0, 50.0, size=n_rows)
    mid = 50_000.0 + np.cumsum(increments)
    df = pd.DataFrame({
        "ts": ts,
        "exchange": pd.Series([exchange] * n_rows, dtype="string"),
        "symbol": pd.Series([symbol] * n_rows, dtype="string"),
        "mid_price": mid,
        "is_valid": True,
    })
    out_dir = features_path(tmp_path, exchange, symbol, dt, interval_ms)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Write using arrow so the schema is parquet-clean.
    table = pa.Table.from_pandas(df, preserve_index=False)
    # Cast exchange/symbol to plain strings so dictionary encoding doesn't
    # leak into label_engine's read path.
    table = table.set_column(
        table.schema.get_field_index("exchange"),
        "exchange",
        table.column("exchange").cast(pa.string()),
    )
    table = table.set_column(
        table.schema.get_field_index("symbol"),
        "symbol",
        table.column("symbol").cast(pa.string()),
    )
    pq.write_table(table, out_dir / "part-00000.parquet")


def _read_labels(tmp_path: Path, exchange: str, symbol: str, dt: datetime,
                 interval_ms: int, horizon_s: int) -> pd.DataFrame:
    out_dir = labels_path(tmp_path, exchange, symbol, dt, interval_ms, horizon_s)
    files = sorted(out_dir.glob("part-*.parquet"))
    if not files:
        return pd.DataFrame()
    # ParquetFile.read() bypasses the dataset layer that would otherwise
    # try to re-derive interval_ms / exchange / etc. from the Hive-style
    # directory names. See src.storage.parquet_store.read_parquet_dir.
    return pa.concat_tables(
        [pq.ParquetFile(str(f)).read() for f in files],
        promote_options="default",
    ).to_pandas()


def test_labels_for_two_intervals_do_not_contaminate_each_other(tmp_path):
    """Writing labels at 100ms and 1000ms must produce two non-overlapping trees."""
    exchange, symbol = "bitfinex", "BTCUSD"
    dt = datetime(2026, 5, 22, tzinfo=timezone.utc)
    cost = {"taker_fee": 10, "half_spread_buffer": 3, "slippage_buffer": 2}

    # Two distinct feature sets so we can fingerprint each label set by row count.
    _write_toy_features(tmp_path, exchange, symbol, dt, interval_ms=100,
                         n_rows=600, seed=1)
    _write_toy_features(tmp_path, exchange, symbol, dt, interval_ms=1000,
                         n_rows=120, seed=2)

    n100 = generate_labels(tmp_path, exchange, symbol, dt,
                            interval_ms=100, horizon_s=5,
                            cost_components_bps=cost)
    n1000 = generate_labels(tmp_path, exchange, symbol, dt,
                             interval_ms=1000, horizon_s=5,
                             cost_components_bps=cost)
    assert n100 > 0 and n1000 > 0

    lab100 = _read_labels(tmp_path, exchange, symbol, dt, 100, 5)
    lab1000 = _read_labels(tmp_path, exchange, symbol, dt, 1000, 5)

    # 1. Row counts come from the respective feature sets, not a mix.
    #    (label engine drops a few tail rows where no future row exists)
    assert len(lab100) == n100
    assert len(lab1000) == n1000
    assert len(lab100) != len(lab1000), \
        "if the two intervals are co-mingling, row counts will look the same"

    # 2. The interval_ms column inside each parquet file tells the truth.
    assert (lab100["interval_ms"] == 100).all()
    assert (lab1000["interval_ms"] == 1000).all()

    # 3. The two output directories are disjoint on disk.
    d100 = labels_path(tmp_path, exchange, symbol, dt, 100, 5)
    d1000 = labels_path(tmp_path, exchange, symbol, dt, 1000, 5)
    assert d100 != d1000
    assert not any(f.name == "part-00000.parquet" and f.parent == d100
                   for f in d1000.glob("**/*"))


def test_label_schema_carries_interval_ms_column():
    """Even outside the path, the row column must be present and int32."""
    names = [f.name for f in LABEL_SCHEMA]
    assert "interval_ms" in names
    assert LABEL_SCHEMA.field("interval_ms").type == pa.int32()
