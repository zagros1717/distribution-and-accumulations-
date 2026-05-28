"""Python port of strategies/trendBreakout.ts."""
from dataclasses import dataclass
from typing import Optional
import pandas as pd

from ..features import atr, adx, ema_stack, cvd_series


@dataclass
class Signal:
    strategy: str; direction: str
    entry: float; stop: float; target: float
    expected_holding_minutes: int
    preferred_entry_mode: str
    partial_tp_r: float; partial_tp_fraction: float
    trail_type: str; trail_atr_mult: float
    reason: str


def trend_breakout_signal(bars1m: pd.DataFrame, bars15m: pd.DataFrame, bars1h: pd.DataFrame, trades: pd.DataFrame, btc_lead: int, cfg: dict) -> Optional[Signal]:
    if len(bars1m) < 60 or len(bars15m) < 30 or len(bars1h) < 50:
        return None
    a15 = adx(bars15m, 14)
    if a15 < cfg.get("min_adx_15m", 25):
        return None
    htf = ema_stack(bars1h)
    if cfg.get("ema_stack_required", True) and htf == 0:
        return None
    direction = "long" if htf == 1 else "short" if htf == -1 else None
    if direction is None:
        return None

    lb = cfg.get("breakout_lookback_min", 5)
    recent = bars1m.iloc[-(lb + 1):-1]
    if len(recent) < lb:
        return None
    recent_high = float(recent["high"].max())
    recent_low = float(recent["low"].min())
    last = bars1m.iloc[-1]
    price = float(last["close"])

    if direction == "long" and not (price > recent_high):
        return None
    if direction == "short" and not (price < recent_low):
        return None

    if cfg.get("require_cvd_new_extreme", True) and not trades.empty:
        end_ts = bars1m.index[-1]
        cvd, _ = cvd_series(trades, end_ts, (lb + 2) * 60_000)
        if len(cvd) >= 4:
            last_cvd = cvd[-1]
            prior = cvd[:-1]
            if direction == "long" and last_cvd < max(prior):
                return None
            if direction == "short" and last_cvd > min(prior):
                return None

    a1 = atr(bars1m, 14)
    stop = recent_low - a1 * 0.2 if direction == "long" else recent_high + a1 * 0.2
    r = abs(price - stop)
    target = price + r * 1.0 if direction == "long" else price - r * 1.0

    return Signal(
        strategy="trend_breakout", direction=direction, entry=price, stop=stop, target=target,
        expected_holding_minutes=18, preferred_entry_mode="depth_aware",
        partial_tp_r=cfg.get("partial_tp_r", 1.0), partial_tp_fraction=cfg.get("partial_tp_fraction", 0.5),
        trail_type="chandelier", trail_atr_mult=cfg.get("chandelier_atr_mult", 2.5),
        reason=f"TREND {direction.upper()} brk={recent_high if direction == 'long' else recent_low:.4f} adx15={a15:.1f}",
    )
