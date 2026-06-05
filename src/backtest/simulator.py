"""
Offline backtest simulator.

Conservative event-driven simulator. It uses only OOS predictions from the
walk-forward trainer and refuses to trade across stale/gappy snapshots.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List

import numpy as np
import pandas as pd


@dataclass
class BacktestConfig:
    latency_ms: int = 250
    taker_fee_bps: float = 10.0
    maker_fee_bps: float = 2.0
    slippage_bps: float = 2.0
    min_confidence: float = 0.60
    max_trades_per_day: int = 500
    cooldown_seconds_after_trade: float = 2.0
    position_size_btc: float = 0.01
    starting_cash_usd: float = 100_000.0
    # Max allowed delay between desired execution time and the next available
    # snapshot. None means infer from the median feature cadence. Trades that
    # need stale snapshots are skipped rather than producing optimistic fills.
    max_snapshot_delay_ms: int | None = None


@dataclass
class Trade:
    ts_signal: pd.Timestamp
    ts_entry: pd.Timestamp
    ts_exit: pd.Timestamp
    direction: int           # +1 long, -1 short
    entry_price: float
    exit_price: float
    size_btc: float
    pnl_gross_usd: float
    pnl_net_usd: float
    fees_usd: float
    return_bps: float
    confidence: float


@dataclass
class BacktestResult:
    config: BacktestConfig
    trades: List[Trade] = field(default_factory=list)
    daily_pnl: pd.DataFrame = field(default_factory=pd.DataFrame)
    summary: dict = field(default_factory=dict)


def _infer_snapshot_tolerance(df: pd.DataFrame, config: BacktestConfig) -> pd.Timedelta:
    if config.max_snapshot_delay_ms is not None:
        return pd.Timedelta(milliseconds=max(int(config.max_snapshot_delay_ms), 1))
    if len(df) < 2:
        return pd.Timedelta(milliseconds=1000)
    diffs = df["ts"].sort_values().diff().dropna()
    if diffs.empty:
        return pd.Timedelta(milliseconds=1000)
    # Twice the median cadence tolerates normal jitter but rejects real gaps.
    return max(diffs.median() * 2, pd.Timedelta(milliseconds=1))


def run_backtest(
    features_df: pd.DataFrame,
    predictions: np.ndarray,
    confidences: np.ndarray,
    horizon_s: int,
    config: BacktestConfig,
) -> BacktestResult:
    """
    Simulate trades from predictions aligned 1:1 with `features_df` rows.

    `predictions`: array of -1/0/+1
    `confidences`: max-class probability per row
    """
    assert len(features_df) == len(predictions) == len(confidences), \
        "features_df, predictions, confidences must align"

    df = features_df.copy()
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.sort_values("ts").reset_index(drop=True)
    df["pred"] = predictions
    df["conf"] = confidences
    if "is_valid" not in df.columns:
        df["is_valid"] = True

    latency_td = pd.Timedelta(milliseconds=config.latency_ms)
    horizon_td = pd.Timedelta(seconds=horizon_s)
    cooldown_td = pd.Timedelta(seconds=config.cooldown_seconds_after_trade)
    max_delay = _infer_snapshot_tolerance(df, config)

    ts_values = df["ts"].values

    def find_next_idx(target_ts: pd.Timestamp) -> int:
        idx = np.searchsorted(ts_values, np.datetime64(target_ts), side="left")
        return int(idx) if idx < len(ts_values) else -1

    def is_fresh(actual_ts: pd.Timestamp, target_ts: pd.Timestamp) -> bool:
        return actual_ts >= target_ts and (actual_ts - target_ts) <= max_delay

    trades: List[Trade] = []
    next_allowed_ts = df["ts"].iloc[0] if not df.empty else pd.Timestamp.utcnow()
    trades_today: dict[pd.Timestamp, int] = {}
    skipped_stale_entry = 0
    skipped_stale_exit = 0
    skipped_invalid_execution = 0

    fee_rate = config.taker_fee_bps / 10000.0
    slip_rate = config.slippage_bps / 10000.0

    for _, row in df.iterrows():
        pred = int(row["pred"])
        if pred == 0:
            continue
        if row["conf"] < config.min_confidence:
            continue
        if not bool(row["is_valid"]):
            continue
        if row["ts"] < next_allowed_ts:
            continue
        day_key = row["ts"].normalize()
        if trades_today.get(day_key, 0) >= config.max_trades_per_day:
            continue

        entry_target = row["ts"] + latency_td
        entry_idx = find_next_idx(entry_target)
        if entry_idx < 0:
            skipped_stale_entry += 1
            continue
        entry_row = df.iloc[entry_idx]
        if not is_fresh(entry_row["ts"], entry_target):
            skipped_stale_entry += 1
            continue
        if not bool(entry_row.get("is_valid", True)):
            skipped_invalid_execution += 1
            continue

        entry_mid = float(entry_row["mid_price"])
        entry_spread = float(entry_row["spread"]) if pd.notna(entry_row["spread"]) else 0.0
        entry_price = entry_mid + (pred * entry_spread / 2.0)
        entry_price *= (1.0 + pred * slip_rate)

        exit_target = entry_row["ts"] + horizon_td
        exit_idx = find_next_idx(exit_target)
        if exit_idx < 0 or exit_idx <= entry_idx:
            skipped_stale_exit += 1
            continue
        exit_row = df.iloc[exit_idx]
        if not is_fresh(exit_row["ts"], exit_target):
            skipped_stale_exit += 1
            continue
        if not bool(exit_row.get("is_valid", True)):
            skipped_invalid_execution += 1
            continue

        exit_mid = float(exit_row["mid_price"])
        exit_spread = float(exit_row["spread"]) if pd.notna(exit_row["spread"]) else 0.0
        exit_price = exit_mid - (pred * exit_spread / 2.0)
        exit_price *= (1.0 - pred * slip_rate)

        size = config.position_size_btc
        notional_in = size * entry_price
        notional_out = size * exit_price
        gross = pred * (exit_price - entry_price) * size
        fees = (notional_in + notional_out) * fee_rate
        net = gross - fees
        ret_bps = (net / max(notional_in, 1e-12)) * 10000.0

        trades.append(Trade(
            ts_signal=row["ts"], ts_entry=entry_row["ts"], ts_exit=exit_row["ts"],
            direction=pred, entry_price=entry_price, exit_price=exit_price,
            size_btc=size, pnl_gross_usd=gross, pnl_net_usd=net,
            fees_usd=fees, return_bps=ret_bps, confidence=float(row["conf"]),
        ))
        trades_today[day_key] = trades_today.get(day_key, 0) + 1
        next_allowed_ts = exit_row["ts"] + cooldown_td

    trades_df = pd.DataFrame([asdict(t) for t in trades])
    summary = {
        "n_trades": len(trades),
        "n_long": int(sum(1 for t in trades if t.direction == 1)),
        "n_short": int(sum(1 for t in trades if t.direction == -1)),
        "gross_pnl_usd": float(sum(t.pnl_gross_usd for t in trades)),
        "net_pnl_usd": float(sum(t.pnl_net_usd for t in trades)),
        "fees_usd": float(sum(t.fees_usd for t in trades)),
        "win_rate": float(np.mean([t.pnl_net_usd > 0 for t in trades])) if trades else 0.0,
        "avg_return_bps": float(np.mean([t.return_bps for t in trades])) if trades else 0.0,
        "max_snapshot_delay_ms": int(max_delay / pd.Timedelta(milliseconds=1)),
        "skipped_stale_entry": skipped_stale_entry,
        "skipped_stale_exit": skipped_stale_exit,
        "skipped_invalid_execution": skipped_invalid_execution,
    }

    daily_pnl = pd.DataFrame()
    if not trades_df.empty:
        trades_df["day"] = pd.to_datetime(trades_df["ts_exit"], utc=True).dt.normalize()
        daily_pnl = trades_df.groupby("day").agg(
            n_trades=("pnl_net_usd", "size"),
            net_pnl=("pnl_net_usd", "sum"),
            gross_pnl=("pnl_gross_usd", "sum"),
            fees=("fees_usd", "sum"),
            avg_bps=("return_bps", "mean"),
        ).reset_index()

        cum = trades_df["pnl_net_usd"].cumsum()
        max_dd = float((cum.cummax() - cum).max()) if len(cum) else 0.0
        equity = config.starting_cash_usd + cum
        returns = trades_df["pnl_net_usd"] / config.starting_cash_usd
        sharpe = float(returns.mean() / returns.std() * np.sqrt(252 * 86400 / max(horizon_s, 1))) \
                  if returns.std() > 0 else 0.0
        summary["max_drawdown_usd"] = max_dd
        summary["max_drawdown_bps"] = float(max_dd / config.starting_cash_usd * 10000.0)
        summary["sharpe_approx"] = sharpe
        summary["final_equity_usd"] = float(equity.iloc[-1]) if len(equity) else config.starting_cash_usd

    return BacktestResult(config=config, trades=trades, daily_pnl=daily_pnl, summary=summary)


def run_oos_backtest(
    oos_predictions_path: str | "Path",
    features_df: pd.DataFrame,
    horizon_s: int,
    config: BacktestConfig,
) -> BacktestResult:
    """Run a backtest using only walk-forward validation-window predictions."""
    oos = pd.read_parquet(str(oos_predictions_path))
    oos["ts"] = pd.to_datetime(oos["ts"], utc=True)
    oos = oos.sort_values("ts").drop_duplicates(subset=["ts"], keep="last")

    df = features_df.copy()
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.sort_values("ts").drop_duplicates(subset=["ts"], keep="last").reset_index(drop=True)

    merged = df.merge(
        oos[["ts", "prob_short", "prob_flat", "prob_long"]],
        on="ts", how="inner",
    ).sort_values("ts").reset_index(drop=True)

    if merged.empty:
        return BacktestResult(config=config, trades=[], daily_pnl=pd.DataFrame(), summary={"n_trades": 0})

    proba = merged[["prob_short", "prob_flat", "prob_long"]].values
    pred = np.argmax(proba, axis=1) - 1
    conf = proba.max(axis=1)
    if config.min_confidence > 0:
        pred = np.where(conf >= config.min_confidence, pred, 0)

    feat_cols = [c for c in merged.columns if c not in ("prob_short", "prob_flat", "prob_long")]
    return run_backtest(merged[feat_cols], pred, conf, horizon_s=horizon_s, config=config)
