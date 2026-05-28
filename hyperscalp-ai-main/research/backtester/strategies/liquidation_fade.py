"""Python port of strategies/liquidationFade.ts."""
from dataclasses import dataclass
from typing import Optional
import pandas as pd

from ..features import atr


@dataclass
class Signal:
    strategy: str; direction: str
    entry: float; stop: float; target: float
    expected_holding_minutes: int
    preferred_entry_mode: str
    reason: str


def liquidation_fade_signal(bars1m: pd.DataFrame, trades: pd.DataFrame, liquidations: pd.DataFrame, coin: str, cfg: dict) -> Optional[Signal]:
    if len(bars1m) < 30 or liquidations.empty:
        return None
    end_ts = bars1m.index[-1]
    window = liquidations[(liquidations["coin"] == coin) & (liquidations["ts"] >= end_ts - pd.Timedelta(seconds=5)) & (liquidations["ts"] <= end_ts)]
    if window.empty:
        return None
    long_usd = float(window.loc[window["side"] == "long", "usd"].sum())
    short_usd = float(window.loc[window["side"] == "short", "usd"].sum())
    net = abs(long_usd - short_usd)
    if net < cfg.get("min_liquidation_usd_5s", 250_000):
        return None

    direction = "long" if long_usd > short_usd else "short"

    lb = cfg.get("sweep_lookback_min", 5)
    recent = bars1m.iloc[-(lb + 1):-1]
    if recent.empty:
        return None
    prev_high = float(recent["high"].max())
    prev_low = float(recent["low"].min())
    last = bars1m.iloc[-1]
    swept = (last["low"] < prev_low) if direction == "long" else (last["high"] > prev_high)
    if not swept:
        return None

    if cfg.get("require_aggressor_flip", True) and not trades.empty:
        # Last 10s aggressor imbalance
        sub = trades[trades["ts"] >= end_ts - pd.Timedelta(seconds=10)]
        if not sub.empty:
            buy = sub.loc[sub["is_buy"], "sz"].sum()
            sell = sub.loc[~sub["is_buy"], "sz"].sum()
            tot = buy + sell
            imb = (buy - sell) / tot if tot > 0 else 0
            if direction == "long" and imb < 0.10:
                return None
            if direction == "short" and imb > -0.10:
                return None

    price = float(last["close"])
    a1 = atr(bars1m, 14)
    stop = prev_low - a1 * 0.3 if direction == "long" else prev_high + a1 * 0.3
    r = abs(price - stop)
    target = price + r * cfg.get("rr_target", 1.5) if direction == "long" else price - r * cfg.get("rr_target", 1.5)

    return Signal(
        strategy="liquidation_fade", direction=direction, entry=price, stop=stop, target=target,
        expected_holding_minutes=cfg.get("max_holding_minutes", 8),
        preferred_entry_mode="ioc",
        reason=f"LIQ_FADE {direction.upper()} long_liq=${long_usd:.0f} short_liq=${short_usd:.0f}",
    )
