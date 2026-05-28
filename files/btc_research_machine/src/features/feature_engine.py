"""
Feature engine.

Given a stream of normalized BookEvents and the snapshot table for an
(exchange, symbol, day), produce a feature row per `interval_ms` bucket.

CRITICAL anti-leakage contract:
  - The feature at bucket ending at time T uses ONLY events with
    event_time <= T. We bucket events by floor(event_time / interval).
  - Rolling-window features at T use closed-left, closed-right windows
    [T - window, T]. We use pandas rolling with `closed='right'`.
  - No feature consults future rows. Tests verify this.

Output: one parquet directory per (interval_ms, exchange, symbol, date).
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.storage.parquet_store import (
    normalized_path, snapshots_path, features_path, read_parquet_dir, ParquetWriter
)
from src.utils.logging import logger


# Schema is dynamic (depth_levels configurable), so we build it in-flight.
def _build_feature_schema(depth_levels: List[int]) -> pa.Schema:
    fields = [
        pa.field("ts", pa.timestamp("us", tz="UTC")),
        pa.field("exchange", pa.string()),
        pa.field("symbol", pa.string()),

        # L1
        pa.field("best_bid", pa.float64()),
        pa.field("best_ask", pa.float64()),
        pa.field("mid_price", pa.float64()),
        pa.field("spread", pa.float64()),
        pa.field("relative_spread", pa.float64()),
        pa.field("last_trade_price", pa.float64()),
        pa.field("last_trade_size", pa.float64()),

        # Microprice
        pa.field("microprice", pa.float64()),
        pa.field("microprice_minus_mid", pa.float64()),

        # L3 order flow (per-interval counts)
        pa.field("new_bid_orders_count", pa.int64()),
        pa.field("new_ask_orders_count", pa.int64()),
        pa.field("cancel_bid_orders_count", pa.int64()),
        pa.field("cancel_ask_orders_count", pa.int64()),
        pa.field("modify_bid_orders_count", pa.int64()),
        pa.field("modify_ask_orders_count", pa.int64()),
        pa.field("bid_cancel_rate", pa.float64()),
        pa.field("ask_cancel_rate", pa.float64()),
        pa.field("large_order_added_bid_count", pa.int64()),
        pa.field("large_order_added_ask_count", pa.int64()),
        pa.field("large_order_cancelled_bid_count", pa.int64()),
        pa.field("large_order_cancelled_ask_count", pa.int64()),

        # Trades
        pa.field("buy_trade_volume", pa.float64()),
        pa.field("sell_trade_volume", pa.float64()),
        pa.field("trade_count", pa.int64()),
        pa.field("trade_imbalance", pa.float64()),
        pa.field("vwap", pa.float64()),

        # Volatility / momentum
        pa.field("mid_return_1s", pa.float64()),
        pa.field("mid_return_5s", pa.float64()),
        pa.field("mid_return_10s", pa.float64()),
        pa.field("realized_vol_10s", pa.float64()),
        pa.field("realized_vol_60s", pa.float64()),

        # Book-health flag — set False when snapshot was corrupt
        pa.field("is_valid", pa.bool_()),
    ]
    for n in depth_levels:
        fields += [
            pa.field(f"bid_depth_{n}", pa.float64()),
            pa.field(f"ask_depth_{n}", pa.float64()),
            pa.field(f"order_book_imbalance_{n}", pa.float64()),
        ]
    return pa.schema(fields)


def generate_features(
    data_root: str | Path,
    exchange: str,
    symbol: str,
    date: datetime,
    interval_ms: int = 1000,
    depth_levels: List[int] = (1, 5, 10),
    large_order_threshold_btc: float = 5.0,
) -> int:
    """
    Build the feature dataframe for one day and persist as parquet.
    Returns row count.
    """
    snap_dir = snapshots_path(data_root, exchange, symbol, date, interval_ms)
    evt_dir = normalized_path(data_root, exchange, symbol, date)
    snap_tbl = read_parquet_dir(snap_dir)
    evt_tbl = read_parquet_dir(evt_dir)
    if snap_tbl.num_rows == 0:
        logger.warning(f"features: no snapshots at {snap_dir}")
        return 0

    snaps = snap_tbl.to_pandas()
    events = evt_tbl.to_pandas() if evt_tbl.num_rows else pd.DataFrame()

    # Index snapshots on ts so we can align bucket counts.
    snaps = snaps.sort_values("ts").reset_index(drop=True)
    interval_td = pd.Timedelta(milliseconds=interval_ms)
    snaps["ts"] = pd.to_datetime(snaps["ts"], utc=True)

    # ---- L3 order-flow counts per bucket ---------------------------------
    if not events.empty:
        events["event_time"] = pd.to_datetime(events["event_time"], utc=True)
        # Bucket each event by the snapshot timestamp it belongs to (right-closed).
        events["bucket"] = events["event_time"].dt.ceil(f"{interval_ms}ms")

        def is_large(s):
            return (s.fillna(0) >= large_order_threshold_btc).astype(int)

        events["is_large"] = is_large(events["size"])

        flows = events.groupby(["bucket", "event_type", "side"]).size().unstack(fill_value=0)
        # The pivot will be multi-indexed; reshape to flat columns.
        flow_counts = events.pivot_table(
            index="bucket",
            columns=["event_type", "side"],
            values="event_time", aggfunc="count", fill_value=0,
        )
        # Trade aggregates (separate because trade rows have aggressor_side, not side).
        trades = events[events["event_type"] == "trade"].copy()
        trades["aggressor_side"] = trades["aggressor_side"].fillna("unknown")

        trade_buy_vol = trades[trades["aggressor_side"] == "bid"].groupby("bucket")["trade_size"].sum()
        trade_sell_vol = trades[trades["aggressor_side"] == "ask"].groupby("bucket")["trade_size"].sum()
        trade_count = trades.groupby("bucket").size()
        trade_vwap = (
            (trades["trade_price"] * trades["trade_size"]).groupby(trades["bucket"]).sum()
            / trades.groupby("bucket")["trade_size"].sum()
        )
        last_trade = trades.groupby("bucket").agg(
            last_trade_price=("trade_price", "last"),
            last_trade_size=("trade_size", "last"),
        )

        # Large order add/cancel counts
        large_evts = events[events["is_large"] == 1]
        large_add_bid = large_evts[(large_evts["event_type"] == "add") & (large_evts["side"] == "bid")].groupby("bucket").size()
        large_add_ask = large_evts[(large_evts["event_type"] == "add") & (large_evts["side"] == "ask")].groupby("bucket").size()
        large_cxl_bid = large_evts[(large_evts["event_type"] == "cancel") & (large_evts["side"] == "bid")].groupby("bucket").size()
        large_cxl_ask = large_evts[(large_evts["event_type"] == "cancel") & (large_evts["side"] == "ask")].groupby("bucket").size()
    else:
        flow_counts = pd.DataFrame()
        trade_buy_vol = pd.Series(dtype=float)
        trade_sell_vol = pd.Series(dtype=float)
        trade_count = pd.Series(dtype=int)
        trade_vwap = pd.Series(dtype=float)
        last_trade = pd.DataFrame(columns=["last_trade_price", "last_trade_size"])
        large_add_bid = large_add_ask = large_cxl_bid = large_cxl_ask = pd.Series(dtype=int)

    # ---- assemble feature frame -----------------------------------------
    df = pd.DataFrame(index=snaps["ts"])
    df["exchange"] = exchange
    df["symbol"] = symbol
    df["best_bid"] = snaps["best_bid"].values
    df["best_ask"] = snaps["best_ask"].values
    df["mid_price"] = snaps["mid_price"].values
    df["spread"] = snaps["spread"].values
    df["relative_spread"] = (df["spread"] / df["mid_price"]).replace([np.inf, -np.inf], np.nan)
    df["is_valid"] = snaps["is_valid"].values

    # Microprice — uses size at best level
    bid_sz = snaps["bid_size_1"].values
    ask_sz = snaps["ask_size_1"].values
    denom = bid_sz + ask_sz
    with np.errstate(invalid="ignore", divide="ignore"):
        microprice = np.where(
            denom > 0,
            (df["best_bid"].values * ask_sz + df["best_ask"].values * bid_sz) / denom,
            np.nan,
        )
    df["microprice"] = microprice
    df["microprice_minus_mid"] = df["microprice"] - df["mid_price"]

    # Depth + imbalance
    for n in depth_levels:
        bid_col = f"bid_size_{n}"
        ask_col = f"ask_size_{n}"
        if bid_col not in snaps.columns or ask_col not in snaps.columns:
            df[f"bid_depth_{n}"] = np.nan
            df[f"ask_depth_{n}"] = np.nan
            df[f"order_book_imbalance_{n}"] = np.nan
            continue
        df[f"bid_depth_{n}"] = snaps[bid_col].values
        df[f"ask_depth_{n}"] = snaps[ask_col].values
        denom_n = snaps[bid_col].values + snaps[ask_col].values
        with np.errstate(invalid="ignore", divide="ignore"):
            df[f"order_book_imbalance_{n}"] = np.where(denom_n > 0,
                snaps[bid_col].values / denom_n, np.nan)

    # Reindex flow series to snapshot timestamps
    def _aligned(series_or_df, col=None, fill=0):
        if isinstance(series_or_df, pd.Series):
            return series_or_df.reindex(df.index, fill_value=fill).values
        if col is None or col not in series_or_df.columns:
            return np.full(len(df), fill, dtype=float)
        return series_or_df[col].reindex(df.index, fill_value=fill).values

    def _flow(event_type: str, side: str):
        if flow_counts.empty:
            return np.zeros(len(df), dtype=int)
        key = (event_type, side)
        if key not in flow_counts.columns:
            return np.zeros(len(df), dtype=int)
        return flow_counts[key].reindex(df.index, fill_value=0).astype(int).values

    df["new_bid_orders_count"] = _flow("add", "bid")
    df["new_ask_orders_count"] = _flow("add", "ask")
    df["cancel_bid_orders_count"] = _flow("cancel", "bid")
    df["cancel_ask_orders_count"] = _flow("cancel", "ask")
    df["modify_bid_orders_count"] = _flow("modify", "bid")
    df["modify_ask_orders_count"] = _flow("modify", "ask")

    new_bid = df["new_bid_orders_count"].values
    new_ask = df["new_ask_orders_count"].values
    cxl_bid = df["cancel_bid_orders_count"].values
    cxl_ask = df["cancel_ask_orders_count"].values
    with np.errstate(invalid="ignore", divide="ignore"):
        df["bid_cancel_rate"] = np.where(new_bid + cxl_bid > 0, cxl_bid / (new_bid + cxl_bid), np.nan)
        df["ask_cancel_rate"] = np.where(new_ask + cxl_ask > 0, cxl_ask / (new_ask + cxl_ask), np.nan)

    df["large_order_added_bid_count"] = _aligned(large_add_bid).astype(int)
    df["large_order_added_ask_count"] = _aligned(large_add_ask).astype(int)
    df["large_order_cancelled_bid_count"] = _aligned(large_cxl_bid).astype(int)
    df["large_order_cancelled_ask_count"] = _aligned(large_cxl_ask).astype(int)

    df["buy_trade_volume"] = _aligned(trade_buy_vol, fill=0.0)
    df["sell_trade_volume"] = _aligned(trade_sell_vol, fill=0.0)
    df["trade_count"] = _aligned(trade_count).astype(int)
    total_trade_vol = df["buy_trade_volume"].values + df["sell_trade_volume"].values
    with np.errstate(invalid="ignore", divide="ignore"):
        df["trade_imbalance"] = np.where(total_trade_vol > 0,
            (df["buy_trade_volume"].values - df["sell_trade_volume"].values) / total_trade_vol,
            np.nan)
    df["vwap"] = _aligned(trade_vwap, fill=np.nan)
    if not last_trade.empty:
        df["last_trade_price"] = last_trade["last_trade_price"].reindex(df.index).ffill().values
        df["last_trade_size"] = last_trade["last_trade_size"].reindex(df.index).fillna(0.0).values
    else:
        df["last_trade_price"] = np.nan
        df["last_trade_size"] = 0.0

    # ---- Volatility / momentum (closed-right rolling on PAST data only) --
    #
    # The number of rows per second depends on interval_ms. A "5-second" return
    # at 1000ms intervals is shift(5); at 100ms intervals it is shift(50). We
    # convert from seconds to rows so feature semantics stay constant as
    # interval_ms changes.
    #
    # We ceil(rows_per_second * seconds) so windows are at least one row even
    # when the cadence is slower than the requested window (e.g. a 1-second
    # return at 5000ms intervals collapses to shift(1)).
    def _rows_for_seconds(seconds: int | float) -> int:
        rows = int(np.ceil(seconds * 1000.0 / interval_ms))
        return max(rows, 1)

    n_1s = _rows_for_seconds(1)
    n_5s = _rows_for_seconds(5)
    n_10s = _rows_for_seconds(10)
    n_60s = _rows_for_seconds(60)
    # Realized-vol min_periods: scale with the window so we don't emit a vol
    # number from one or two observations (which would be noise, not signal).
    rv_10s_min = max(2, n_10s // 5)
    rv_60s_min = max(2, n_60s // 6)

    mid = df["mid_price"]
    log_mid = np.log(mid.replace(0, np.nan))
    df["mid_return_1s"] = log_mid - log_mid.shift(n_1s)
    df["mid_return_5s"] = log_mid - log_mid.shift(n_5s)
    df["mid_return_10s"] = log_mid - log_mid.shift(n_10s)
    # Realized vol = sqrt of sum of squared one-step log returns within the
    # past window. "One step" here is one interval_ms, so a 10-second realized
    # vol at 1000ms intervals is the same as a 100-step rolling sum at 100ms.
    r1 = log_mid - log_mid.shift(1)  # one-interval log return (interval-step)
    df["realized_vol_10s"] = np.sqrt((r1 ** 2).rolling(n_10s, min_periods=rv_10s_min).sum())
    df["realized_vol_60s"] = np.sqrt((r1 ** 2).rolling(n_60s, min_periods=rv_60s_min).sum())

    # ---- Reorder + write -------------------------------------------------
    df = df.reset_index().rename(columns={"index": "ts"})
    df["ts"] = pd.to_datetime(df["ts"], utc=True)

    schema = _build_feature_schema(list(depth_levels))
    out_dir = features_path(data_root, exchange, symbol, date, interval_ms)
    writer = ParquetWriter(out_dir, schema, flush_rows=50_000, flush_seconds=30)
    writer.write(df.to_dict(orient="records"))
    writer.close()
    logger.info(f"features: {exchange}/{symbol} {date.date()} interval={interval_ms}ms -> {len(df)} rows")
    return len(df)
