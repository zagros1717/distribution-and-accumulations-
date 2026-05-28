"""
Label engine.

For each feature row at time T, compute:

    future_mid_T+h = mid_price at the closest feature row whose ts >= T + h
    future_return  = future_mid_T+h / mid_price_T - 1

Classification target:

    +1 if future_return > +threshold
    -1 if future_return < -threshold
     0 otherwise

`threshold` must reflect the FULL round-trip cost of trying to capture the
move — i.e. the same costs the backtester applies. A taker round trip pays
taker_fee on entry AND on exit, plus half-spread + slippage on each leg.

Therefore:

    threshold_bps = 2 * (taker_fee_bps + half_spread_buffer_bps + slippage_buffer_bps)

A +1 label means "the move was big enough to overcome the round trip" —
not just "the move was bigger than one-way costs." Using a one-way threshold
produced systematically optimistic labels that backtests then fail to deliver.

CRITICAL: labels are intentionally a FUTURE function of price. They are the
only thing in this codebase that may peek forward. They must never be used
as features.
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
    pa.field("future_mid_price", pa.float64()),
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
    Compute the move-size threshold (in bps) a label must clear.

    Returns the ROUND-TRIP cost by default — taker fee + half spread + slippage,
    applied on both entry and exit legs. This matches the backtester's
    cost model.

    Pass round_trip=False only if you really mean a one-way threshold (e.g.
    for diagnostic comparisons).
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
    Build labels for one (exchange, symbol, date, interval, horizon) and write
    parquet under data/labels/interval_ms=.../horizon_s=.../... .
    Returns row count.
    """
    in_dir = features_path(data_root, exchange, symbol, date, interval_ms)
    feat = read_parquet_dir(in_dir)
    if feat.num_rows == 0:
        logger.warning(f"labels: no features at {in_dir}")
        return 0
    df = feat.to_pandas()
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.sort_values("ts").reset_index(drop=True)

    # Look forward by horizon_s using a timestamp join.
    df["future_ts"] = df["ts"] + pd.Timedelta(seconds=horizon_s)
    right = df[["ts", "mid_price"]].rename(columns={"ts": "future_ts", "mid_price": "future_mid_price"})
    merged = pd.merge_asof(
        df[["ts", "mid_price", "future_ts"]].sort_values("future_ts"),
        right.sort_values("future_ts"),
        on="future_ts",
        direction="forward",
        tolerance=pd.Timedelta(seconds=horizon_s),  # require we found one within +horizon_s
    )
    merged = merged.sort_values("ts").reset_index(drop=True)

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
        "future_mid_price": merged["future_mid_price"].astype(float),
        "future_return": future_return.astype(float),
        "threshold_bps": float(threshold_bps),
        "label": label,
        # XGBoost-friendly remap: -1 -> 0, 0 -> 1, +1 -> 2
        "label_class": (label + 1).astype(np.int8),
    })

    # Drop rows where we couldn't find a future point (end of day).
    out = out.dropna(subset=["future_mid_price"])

    out_dir = labels_path(data_root, exchange, symbol, date, interval_ms, horizon_s)
    writer = ParquetWriter(out_dir, LABEL_SCHEMA, flush_rows=50_000, flush_seconds=30)
    writer.write(out.to_dict(orient="records"))
    writer.close()
    logger.info(
        f"labels: {exchange}/{symbol} {date.date()} interval={interval_ms}ms "
        f"horizon={horizon_s}s threshold={threshold_bps:.1f}bps (round-trip) "
        f"-> {len(out)} rows, class balance "
        f"(-1/0/+1) = ({(out['label']==-1).sum()},{(out['label']==0).sum()},{(out['label']==1).sum()})"
    )
    return len(out)
