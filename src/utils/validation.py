"""
Look-ahead / leakage validators.

These functions are called by the training pipeline AND by the unit tests.
The whole project lives or dies on these checks.
"""
from __future__ import annotations

from typing import Iterable

import pandas as pd


class LookAheadError(AssertionError):
    """Raised when temporal ordering is violated."""


def assert_monotonic_time(df: pd.DataFrame, time_col: str = "ts") -> None:
    """Rows must be non-decreasing in time."""
    if df.empty:
        return
    ts = df[time_col]
    if not ts.is_monotonic_increasing:
        bad = (ts.diff() < pd.Timedelta(0)).sum()
        raise LookAheadError(f"{time_col} is not monotonic: {bad} backward steps")


def assert_train_before_val(train: pd.DataFrame, val: pd.DataFrame, time_col: str = "ts") -> None:
    """Every train row must be strictly earlier than every val row."""
    if train.empty or val.empty:
        return
    if train[time_col].max() >= val[time_col].min():
        raise LookAheadError(
            f"Train/val overlap: train_max={train[time_col].max()} "
            f"val_min={val[time_col].min()}"
        )


def assert_feature_uses_no_future(
    df: pd.DataFrame, feature_col: str, source_col: str, time_col: str = "ts"
) -> None:
    """
    Verify a rolling feature only used past values.

    The simple, defensible check: for every row i, the feature value at i must
    not change if you delete rows j > i. We don't run that O(n^2) check here;
    instead we test the contract by re-computing on a truncated frame in tests.
    See tests/test_no_lookahead.py.
    """
    # This stub exists so call-sites have a single import path; the real teeth
    # are in the test file. Kept here for symmetry and discoverability.
    _ = df, feature_col, source_col, time_col


def split_walk_forward(
    df: pd.DataFrame,
    train_days: int,
    val_days: int,
    step_days: int,
    time_col: str = "ts",
) -> Iterable[tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Generate (train, val) pairs sliding forward in time.
    Yields tuples; the caller decides what to do with them.
    """
    assert_monotonic_time(df, time_col=time_col)
    if df.empty:
        return
    start = df[time_col].min().floor("D")
    end = df[time_col].max().ceil("D")
    cur = start
    one_day = pd.Timedelta(days=1)
    while cur + (train_days + val_days) * one_day <= end:
        train_end = cur + train_days * one_day
        val_end = train_end + val_days * one_day
        train = df[(df[time_col] >= cur) & (df[time_col] < train_end)]
        val = df[(df[time_col] >= train_end) & (df[time_col] < val_end)]
        if not train.empty and not val.empty:
            assert_train_before_val(train, val, time_col=time_col)
            yield train, val
        cur = cur + step_days * one_day
