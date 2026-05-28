"""
Vectorized event-driven backtester. Iterates 1m bars in chronological order,
calls each enabled strategy at each bar, simulates entry (taker or maker),
manages exits (TP / SL / partial / trail / time stop), applies the cost model,
and emits a per-trade DataFrame.

This is the truth-source for evaluating any strategy change before live.
"""
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Dict
import pandas as pd
import numpy as np

from .cost_model import CostConfig, round_trip_cost_bps, passes_edge_filter
from .features import atr


@dataclass
class TradeRecord:
    coin: str; strategy: str; direction: str
    entry_ts: pd.Timestamp; exit_ts: Optional[pd.Timestamp]
    entry_px: float; exit_px: Optional[float]
    stop: float; target: float
    size_usd: float
    pnl_usd: float
    fees_usd: float
    r_multiple: float
    reason_open: str
    reason_close: str


@dataclass
class BacktestConfig:
    starting_equity: float = 1000.0
    vol_target_usd_per_trade: float = 25.0
    max_open: int = 3
    cost: CostConfig = field(default_factory=CostConfig)
    enforce_edge_filter: bool = True
    time_stop_min: int = 25
    time_stop_min_progress_r: float = 0.4


def simulate(
    coin_data: dict, coin: str, signal_fn: Callable, signal_kwargs: dict,
    cfg: BacktestConfig,
) -> pd.DataFrame:
    """signal_fn must return either None or a Signal-like dataclass with the v6 fields."""
    bars = coin_data[coin]["bars"]
    trades = coin_data[coin]["trades"]
    funding = coin_data[coin]["funding"]
    liquidations = coin_data.get("_liquidations", pd.DataFrame())

    open_pos: Optional[dict] = None
    records: List[TradeRecord] = []
    equity = cfg.starting_equity

    for i in range(60, len(bars)):
        ts = bars.index[i]
        win = bars.iloc[max(0, i - 200):i + 1]
        # Manage open position first
        if open_pos:
            cur_px = float(bars["close"].iloc[i])
            sl_hit = (cur_px <= open_pos["stop"]) if open_pos["direction"] == "long" else (cur_px >= open_pos["stop"])
            tp_hit = (cur_px >= open_pos["target"]) if open_pos["direction"] == "long" else (cur_px <= open_pos["target"])
            age_min = (ts - open_pos["entry_ts"]).total_seconds() / 60
            r_dist = abs(open_pos["entry_px"] - open_pos["stop"])
            progress_r = (cur_px - open_pos["entry_px"]) / r_dist if open_pos["direction"] == "long" else (open_pos["entry_px"] - cur_px) / r_dist
            time_stop = age_min >= cfg.time_stop_min and progress_r < cfg.time_stop_min_progress_r

            if sl_hit or tp_hit or time_stop:
                exit_px = open_pos["target"] if tp_hit else (open_pos["stop"] if sl_hit else cur_px)
                pnl = (exit_px - open_pos["entry_px"]) * (open_pos["size_usd"] / open_pos["entry_px"]) * (1 if open_pos["direction"] == "long" else -1)
                rv = open_pos["realized_vol"]
                cost_bps = round_trip_cost_bps(cfg.cost, open_pos["entry_is_maker"], False, rv)
                fees = open_pos["size_usd"] * 2 * cost_bps / 10000
                net = pnl - fees
                r_mult = net / max(1.0, r_dist * (open_pos["size_usd"] / open_pos["entry_px"]))
                records.append(TradeRecord(
                    coin=coin, strategy=open_pos["strategy"], direction=open_pos["direction"],
                    entry_ts=open_pos["entry_ts"], exit_ts=ts,
                    entry_px=open_pos["entry_px"], exit_px=float(exit_px),
                    stop=open_pos["stop"], target=open_pos["target"],
                    size_usd=open_pos["size_usd"], pnl_usd=net, fees_usd=fees,
                    r_multiple=r_mult, reason_open=open_pos["reason"],
                    reason_close="tp" if tp_hit else "sl" if sl_hit else "time_stop",
                ))
                equity += net
                open_pos = None

        if open_pos is not None:
            continue

        # Evaluate the strategy at this bar
        funding_bps = 0.0
        if not funding.empty:
            f_now = funding[funding["ts"] <= ts]
            if not f_now.empty:
                funding_bps = float(f_now["funding"].iloc[-1]) * 10000

        kwargs = dict(signal_kwargs)
        kwargs["funding_bps"] = funding_bps
        # Strategy fn signatures vary; call adaptively
        try:
            if signal_fn.__name__ == "range_mr_signal":
                sig = signal_fn(win, trades, funding_bps, 0, kwargs.get("cfg", {}))
            elif signal_fn.__name__ == "trend_breakout_signal":
                bars15m = win["close"].resample("15min").ohlc().dropna()
                bars15m["volume"] = win["volume"].resample("15min").sum().reindex(bars15m.index).fillna(0)
                bars1h = win["close"].resample("1h").ohlc().dropna()
                bars1h["volume"] = win["volume"].resample("1h").sum().reindex(bars1h.index).fillna(0)
                bars15m.columns = [c for c in bars15m.columns]
                bars15m = bars15m.rename(columns={"open": "open", "high": "high", "low": "low", "close": "close"})
                bars1h = bars1h.rename(columns={"open": "open", "high": "high", "low": "low", "close": "close"})
                sig = signal_fn(win, bars15m, bars1h, trades, 0, kwargs.get("cfg", {}))
            elif signal_fn.__name__ == "liquidation_fade_signal":
                sig = signal_fn(win, trades, liquidations, coin, kwargs.get("cfg", {}))
            else:
                sig = None
        except Exception:
            sig = None

        if sig is None:
            continue

        # Cost-aware filter
        a1 = atr(win, 14)
        rv = a1 / max(float(win["close"].iloc[-1]), 1)
        entry_is_maker = sig.preferred_entry_mode == "post_only"
        if cfg.enforce_edge_filter:
            ef = passes_edge_filter(cfg.cost, sig.entry, sig.target, entry_is_maker, False, rv)
            if not ef["ok"]:
                continue

        # Vol-targeted size
        atr_pct = max(rv, 1e-6)
        hold_frac = max(1 / 1440, sig.expected_holding_minutes / 1440)
        expected_move = atr_pct * np.sqrt(hold_frac)
        size = cfg.vol_target_usd_per_trade / max(expected_move, 1e-6)
        size = min(size, equity * 5)  # max 5x leverage
        size = min(size, 500.0)
        if size < 12:
            continue

        open_pos = {
            "coin": coin, "strategy": sig.strategy, "direction": sig.direction,
            "entry_ts": ts, "entry_px": sig.entry, "stop": sig.stop, "target": sig.target,
            "size_usd": float(size), "reason": sig.reason,
            "entry_is_maker": entry_is_maker, "realized_vol": rv,
        }

    return pd.DataFrame([r.__dict__ for r in records])
