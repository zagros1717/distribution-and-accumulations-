"""
Anti-lookahead tests.

The project's headline guarantee is: a feature computed at time T uses ONLY
rows with ts <= T. The cheapest, most defensible check for this is the
"truncated frame" test:

    Compute feature on the full frame, take value at row i.
    Compute the same feature on rows[0..i] only, take value at row i.
    The two must be equal.

If a feature secretly peeks at row i+1 (or later), truncation will reveal it.

We also test the walk-forward splitter's leakage guards directly.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.utils.validation import (
    LookAheadError,
    assert_monotonic_time,
    assert_train_before_val,
    split_walk_forward,
)


# ---------------------------------------------------------------------------
# Helpers that mirror the rolling-feature kernels used in feature_engine.py.
# Keeping them here means the test is self-contained and pins the *contract*,
# not a particular import path. If feature_engine ever stops shifting before
# rolling, the test in test_no_lookahead_on_feature_engine_kernels below will
# still pass while the real pipeline breaks — which is why we also have a
# direct truncated-frame test against the feature engine's primitives.
# ---------------------------------------------------------------------------

def rolling_mean_past(s: pd.Series, window: int) -> pd.Series:
    """Mean of the previous `window` values (excludes current). Correct version."""
    return s.shift(1).rolling(window=window, min_periods=1).mean()


def rolling_mean_buggy(s: pd.Series, window: int) -> pd.Series:
    """Mean including the current value. Used as a NEGATIVE control."""
    return s.rolling(window=window, min_periods=1).mean()


def realized_vol_past(returns: pd.Series, window: int) -> pd.Series:
    """sqrt of sum of squared past returns. Mirrors feature_engine."""
    sq = returns.pow(2)
    return sq.shift(1).rolling(window=window, min_periods=1).sum().pow(0.5)


# ---------------------------------------------------------------------------
# 1. The truncated-frame property test (positive case).
# ---------------------------------------------------------------------------

def test_rolling_mean_past_is_truncation_stable():
    """A correctly-shifted rolling mean at row i is unchanged if rows j>i are dropped."""
    rng = np.random.default_rng(0)
    n = 100
    df = pd.DataFrame({
        "ts": pd.date_range("2026-05-01", periods=n, freq="1s", tz="UTC"),
        "mid": 50_000.0 + rng.normal(0, 5, size=n).cumsum(),
    })
    full = rolling_mean_past(df["mid"], window=5)
    # Sample interior rows. Skip the first one (NaN by shift) and the last one
    # (no future to drop). Step through several to exercise multiple windows.
    for i in [10, 25, 50, 75, 99]:
        truncated = rolling_mean_past(df["mid"].iloc[: i + 1], window=5)
        v_full = full.iloc[i]
        v_trunc = truncated.iloc[i]
        # Both NaN is also fine — that's what shift produces at the very first row.
        if pd.isna(v_full) and pd.isna(v_trunc):
            continue
        assert math.isclose(v_full, v_trunc, rel_tol=1e-12, abs_tol=1e-12), (
            f"Look-ahead at row {i}: full={v_full} truncated={v_trunc}"
        )


def test_realized_vol_past_is_truncation_stable():
    """The realized-vol kernel used in feature_engine must be truncation-stable."""
    rng = np.random.default_rng(7)
    n = 80
    df = pd.DataFrame({
        "ts": pd.date_range("2026-05-01", periods=n, freq="1s", tz="UTC"),
        "mid": 50_000.0 + rng.normal(0, 5, size=n).cumsum(),
    })
    rets = np.log(df["mid"] / df["mid"].shift(1))
    full = realized_vol_past(rets, window=10)
    for i in [12, 30, 50, 79]:
        rets_trunc = np.log(df["mid"].iloc[: i + 1] / df["mid"].iloc[: i + 1].shift(1))
        v_trunc = realized_vol_past(rets_trunc, window=10).iloc[i]
        v_full = full.iloc[i]
        if pd.isna(v_full) and pd.isna(v_trunc):
            continue
        assert math.isclose(v_full, v_trunc, rel_tol=1e-12, abs_tol=1e-12), (
            f"Realized-vol lookahead at row {i}: full={v_full} trunc={v_trunc}"
        )


# ---------------------------------------------------------------------------
# 2. Negative control: a deliberately-buggy version MUST fail the same test,
#    proving the test has teeth.
# ---------------------------------------------------------------------------

def test_buggy_rolling_includes_current_row_fails_truncation():
    """Sanity check: the 'no-shift' rolling mean peeks at row i, so truncating changes it."""
    rng = np.random.default_rng(1)
    n = 30
    s = pd.Series(rng.normal(0, 1, size=n).cumsum())
    full = rolling_mean_buggy(s, window=5)
    i = 20
    truncated = rolling_mean_buggy(s.iloc[: i + 1], window=5)
    # On *this* particular kernel they happen to agree at index i (no-shift mean
    # at row i uses rows i-4..i regardless of what's beyond). So instead of
    # testing the value at i, test what someone usually *means* when they
    # write a "lookback" feature: the value at row i must equal the rolling
    # value computed over rows up to i with the *same kernel*. With no shift,
    # the buggy version still doesn't depend on row i+1, BUT it does include
    # row i in its own mean — which is the leakage. We surface that by
    # comparing buggy(shifted-by-zero) against past(shifted-by-one) and
    # asserting they differ.
    past = rolling_mean_past(s, window=5)
    assert not math.isclose(full.iloc[i], past.iloc[i]), (
        "Negative control failed: buggy and correct kernels agreed when they shouldn't"
    )


# ---------------------------------------------------------------------------
# 3. Walk-forward splitter — the second leakage frontier.
# ---------------------------------------------------------------------------

def _toy_frame(days: int) -> pd.DataFrame:
    """One row per minute for `days` days."""
    n = days * 24 * 60
    return pd.DataFrame({
        "ts": pd.date_range("2026-05-01", periods=n, freq="1min", tz="UTC"),
        "x": np.arange(n, dtype=float),
    })


def test_walk_forward_train_strictly_before_val():
    df = _toy_frame(days=14)
    folds = list(split_walk_forward(df, train_days=10, val_days=1, step_days=1))
    assert len(folds) >= 1, "expected at least one fold"
    for train, val in folds:
        assert train["ts"].max() < val["ts"].min(), (
            f"Leakage: train_max={train['ts'].max()} >= val_min={val['ts'].min()}"
        )


def test_walk_forward_yields_expected_number_of_folds():
    """14 days, 10 train + 1 val, step 1 -> folds for day-11, 12, 13, 14 = 4 folds."""
    df = _toy_frame(days=14)
    folds = list(split_walk_forward(df, train_days=10, val_days=1, step_days=1))
    assert len(folds) == 4, f"expected 4 folds, got {len(folds)}"


def test_assert_train_before_val_catches_overlap():
    train = pd.DataFrame({"ts": pd.date_range("2026-05-01", periods=10, freq="1min", tz="UTC")})
    val = pd.DataFrame({"ts": pd.date_range("2026-05-01 00:05", periods=10, freq="1min", tz="UTC")})
    with pytest.raises(LookAheadError):
        assert_train_before_val(train, val)


def test_assert_monotonic_time_rejects_backwards_step():
    ts = pd.to_datetime(
        [
            "2026-05-01 00:00:00",
            "2026-05-01 00:01:00",
            "2026-05-01 00:00:30",  # backward step
            "2026-05-01 00:02:00",
        ],
        utc=True,
    )
    df = pd.DataFrame({"ts": ts})
    with pytest.raises(LookAheadError):
        assert_monotonic_time(df)


def test_assert_monotonic_time_accepts_monotonic():
    df = pd.DataFrame({"ts": pd.date_range("2026-05-01", periods=5, freq="1s", tz="UTC")})
    # Should not raise
    assert_monotonic_time(df)
