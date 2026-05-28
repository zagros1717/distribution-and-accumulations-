"""Python port of strategies/rangeMR.ts. Pure function: bars+trades+state → Signal|None."""
from dataclasses import dataclass
from typing import Optional
import pandas as pd

from ..features import vwap_bands, atr, adx, choppiness


@dataclass
class Signal:
    strategy: str
    direction: str
    entry: float
    stop: float
    target: float
    expected_holding_minutes: int
    preferred_entry_mode: str
    reason: str


def range_mr_signal(bars1m: pd.DataFrame, trades: pd.DataFrame, funding_bps: float, btc_lead: int, cfg: dict) -> Optional[Signal]:
    if len(bars1m) < 60:
        return None

    # Regime: must be RANGE
    a15 = adx(bars1m.tail(60), 14) if len(bars1m) >= 60 else 0
    chop = choppiness(bars1m.tail(60), 14)
    if a15 >= cfg.get("max_adx_15m", 22):
        return None
    if chop < cfg.get("min_choppiness", 60):
        return None

    session = bars1m.tail(60)
    vwap, sigma, dev = vwap_bands(session)
    if sigma == 0:
        return None
    last = bars1m.iloc[-1]
    price = float(last["close"])

    direction = None
    sigma_entry = cfg.get("vwap_sigma_entry", 2.0)
    if dev <= -sigma_entry:
        direction = "long"
    elif dev >= sigma_entry:
        direction = "short"
    if not direction:
        return None

    # Funding skew
    max_skew = cfg.get("max_funding_skew_bps", 20)
    if direction == "long" and funding_bps > max_skew:
        return None
    if direction == "short" and funding_bps < -max_skew:
        return None

    # Absorption proxy: small wick on the side we're entering from
    a1 = atr(bars1m, 14)
    h, l, c, o = float(last["high"]), float(last["low"]), float(last["close"]), float(last["open"])
    if direction == "long" and (c - l) < a1 * 0.3:
        return None
    if direction == "short" and (h - c) < a1 * 0.3:
        return None

    sigma_stop = cfg.get("vwap_sigma_stop", 3.0)
    target = vwap if cfg.get("tp_at_vwap", True) else (price + sigma if direction == "long" else price - sigma)
    stop = price - sigma * sigma_stop if direction == "long" else price + sigma * sigma_stop

    return Signal(
        strategy="range_mr",
        direction=direction, entry=price, stop=stop, target=target,
        expected_holding_minutes=12,
        preferred_entry_mode="post_only" if cfg.get("use_post_only", True) else "ioc",
        reason=f"RANGE_MR {direction.upper()} dev={dev:.2f}σ vwap={vwap:.4f}",
    )
