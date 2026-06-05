"""
Label engine.

For each feature row at time T, compute a cost-aware future-return label at a
strict horizon. Labels are the only stage allowed to look forward, and they are
never used as features.

Reliability rule:
  A row is labelled only when the future snapshot is close to T + horizon. If
  the stream has a data gap, we drop the row instead of silently turning a 5s
  label into a 9-10s label. The allowed delay defaults to one feature interval.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import pyarrow as pa

from src.storage.parquet_store import features_path, labels_path, read_parquet_dir, ParquetWriter
from src.utils.logging import logger


LABEL_SCHEMA = pa.schema([
    pa.field("ts", pa.timestamp("us", tz="UTC")),
    pa.field("exchange", pa.string()),
    pa.field("symbol", pa.string()),
    pa.field("horizon_s", pa.int32()),
    pa.field("interval_ms", pa.int32()),
    pa.field("mid_price", pa.float64()),
    pa.field("future_ts", pa.timestamp("us", tz="UTC")),
    pa.field("future_mid_price", pa.float64()),
    pa.field("actual_horizon_s", pa.float64()),
    pa.field("horizon_error_ms", pa.float64()),
    pa.field("future_return", pa.float64()),
    pa.field("threshold_bps", pa.float64()),
    pa.field("label", pa.int8()),       # -1, 0, +1
    pa.field("label_class", pa.int8()), # 0, 1, 2 — XGBoost-friendly
])


def compute_threshold_bps(
    cost_components_bps: Dict[str, float],
    round_trip: bool = True,
) -> float:
    """
    Compute the move-size threshold in bps a label must clear.

    The default is the full round-trip cost: taker fee + half spread + slippage
    on entry and again on exit. This matches the backtester cost model.
    """
    one_way = (
        cost_components_bps.get("taker_fee", 0)
        + cost_components_bps.get("half_spread_buffer", 0)
        + cost_components_bps.get("slippage_buffer", 0)
    )
    return one_way * (2.0 if round_trip else 1.0)


def generate_labels(
    data_root: str | Path,
    exchange: str,
    symbol: str,
    date: datetime,
    interval_ms: int,
    horizon_s: int,
    cost_components_bps: Dict[str, float],
) -> int:
    """
    Build labels for one (exchange, symbol, date, interval, horizon) partition.

    Rows are dropped when a future price cannot be found within one feature
    interval after the target timestamp. This prevents data gaps from creating
    variable-horizon labels that look predictive but cannot be traded reliably.
    """
    in_dir = features_path(data_root, exchange, symbol, date, interval_ms)
    feat = read_parquet_dir(in_dir)
    if feat.num_rows == 0:
        logger.warning(f"labels: no features at {in_dir}")
        return 0

    df = feat.to_pandas()
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.sort_values("ts").reset_index(drop=True)
    df = df.dropna(subset=["mid_price"])
    df = df[df["mid_price"] > 0].copy()
    if df.empty:
        logger.warning(f"labels: no valid mid prices at {in_dir}")
        return 0

    target_col = "future_ts_target"
    actual_col = "future_ts"
    df[target_col] = df["ts"] + pd.Timedelta(seconds=horizon_s)

    right = df[["ts", "mid_price"]].rename(
        columns={"ts": actual_col, "mid_price": "future_mid_price"}
    )
    tolerance = pd.Timedelta(milliseconds=max(int(interval_ms), 1))
    merged = pd.merge_asof(
        df[["ts", "mid_price", target_col]].sort_values(target_col),
        right.sort_values(actual_col),
        left_on=target_col,
        right_on=actual_col,
        direction="forward",
        tolerance=tolerance,
    ).sort_values("ts").reset_index(drop=True)

    # Drop end-of-day rows and data-gap rows. Labelling them as flat would make
    # the model learn from missing data rather than future movement.
    before_drop = len(merged)
    merged = merged.dropna(subset=["future_mid_price", actual_col]).copy()
    dropped = before_drop - len(merged)

    if merged.empty:
        logger.warning(
            f"labels: no rows survived strict horizon matching at {in_dir} "
            f"horizon={horizon_s}s tolerance={tolerance}"
        )
        return 0

    merged["actual_horizon_s"] = (merged[actual_col] - merged["ts"]).dt.total_seconds()
    merged["horizon_error_ms"] = (
        (merged[actual_col] - merged[target_col]).dt.total_seconds() * 1000.0
    )

    threshold_bps = compute_threshold_bps(cost_components_bps, round_trip=True)
    threshold = threshold_bps / 10000.0
    future_return = merged["future_mid_price"] / merged["mid_price"] - 1.0
    label = np.where(future_return > threshold, 1,
              np.where(future_return < -threshold, -1, 0)).astype(np.int8)

    out = pd.DataFrame({
        "ts": merged["ts"],
        "exchange": exchange,
        "symbol": symbol,
        "horizon_s": np.int32(horizon_s),
        "interval_ms": np.int32(interval_ms),
        "mid_price": merged["mid_price"].astype(float),
        "future_ts": merged[actual_col],
        "future_mid_price": merged["future_mid_price"].astype(float),
        "actual_horizon_s": merged["actual_horizon_s"].astype(float),
        "horizon_error_ms": merged["horizon_error_ms"].astype(float),
        "future_return": future_return.astype(float),
        "threshold_bps": float(threshold_bps),
        "label": label,
        "label_class": (label + 1).astype(np.int8),
    })

    out_dir = labels_path(data_root, exchange, symbol, date, interval_ms, horizon_s)
    writer = ParquetWriter(out_dir, LABEL_SCHEMA, flush_rows=50_000, flush_seconds=30)
    writer.write(out.to_dict(orient="records"))
    writer.close()
    logger.info(
        f"labels: {exchange}/{symbol} {date.date()} interval={interval_ms}ms "
        f"horizon={horizon_s}s tolerance={tolerance} threshold={threshold_bps:.1f}bps "
        f"-> {len(out)} rows ({dropped} dropped for missing/late future rows), "
        f"class balance (-1/0/+1) = "
        f"({(out['label']==-1).sum()},{(out['label']==0).sum()},{(out['label']==1).sum()})"
    )
    return len(out)
