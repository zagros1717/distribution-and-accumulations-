"""
btcalpha.strategy.signal
~~~~~~~~~~~~~~~~~~~~~~~~
Strategy engine: converts model probabilities into trade decisions.

Risk controls:
  - per-timeframe signal threshold
  - per-timeframe confidence threshold
  - neutral-class margin filter
  - trend warning based on fast/slow moving averages
  - cooldown after exits/flips

Important: macro/regime and trend are advisory by default. They are shown in
reasons, but they should not block BTC signals unless explicitly changed later.
"""
from __future__ import annotations

from typing import Dict, Any

import numpy as np
import pandas as pd

from btcalpha.config import get_config, get_logger

log = get_logger("strategy")


def _kelly_fraction(p_win: float, win_loss_ratio: float) -> float:
    if win_loss_ratio <= 0:
        return 0.0
    f = p_win - (1 - p_win) / win_loss_ratio
    return float(np.clip(f, 0.0, 1.0))


class Strategy:
    """Swing/intraday strategy with risk controls and anti-overtrade filters."""

    def __init__(self, timeframe: str | None = None):
        self.cfg = get_config()
        self.scfg = self.cfg["strategy"]
        self.timeframe = timeframe or self.cfg["data"]["timeframes"][0]

    def _tf_value(self, key: str, default: Any) -> Any:
        raw = self.scfg.get(key, default)
        if isinstance(raw, dict):
            return raw.get(self.timeframe, raw.get("default", default))
        return raw

    def _flat_decision(self, raw, final, conf, regime, reasons, price=None, atr=None, regime_score=0.0, trend_strength=None) -> Dict:
        out = {
            "direction": "flat",
            "raw_alpha": float(raw),
            "final_signal": float(final),
            "position": 0.0,
            "confidence": float(conf),
            "regime": regime,
            "regime_score": float(regime_score),
            "stop_loss": None,
            "take_profit": None,
            "price": float(price) if price is not None else None,
            "atr": float(atr) if atr is not None else None,
            "reasons": reasons,
        }
        if trend_strength is not None:
            out["trend_strength"] = float(trend_strength)
        return out

    def decide_row(
        self,
        proba: Dict[str, float],
        regime_label: str,
        regime_score: float,
        price: float,
        atr: float,
        trend_strength: float = 0.0,
    ) -> Dict:
        p_up = float(proba["p_up"])
        p_down = float(proba["p_down"])
        p_neutral = float(proba["p_neutral"])

        raw_alpha = p_up - p_down
        directional_confidence = max(p_up, p_down)
        confidence = max(p_up, p_down, p_neutral)

        reg_mult = self.scfg["regime_multipliers"].get(regime_label, 1.0)
        final_signal = raw_alpha * reg_mult

        min_conf = float(self._tf_value("min_confidence_by_tf", self.scfg.get("min_confidence", 0.45)))
        min_signal = float(self._tf_value("min_abs_signal_by_tf", self.scfg.get("min_abs_signal", 0.05)))
        neutral_margin = float(self._tf_value("neutral_margin_by_tf", self.scfg.get("neutral_margin", 0.02)))
        allow_short = bool(self._tf_value("allow_short_by_tf", self.cfg["backtest"].get("allow_short", True)))
        trend_filter = bool(self.scfg.get("trend_filter", {}).get("enabled", True))
        trend_threshold = float(self._tf_value("trend_filter_threshold_by_tf", 0.0))

        reasons = [
            f"ML: p_up={p_up:.2f} p_down={p_down:.2f} p_neutral={p_neutral:.2f}",
            f"رژیم={regime_label} (ضریب {reg_mult})",
            f"threshold={min_signal:.3f} conf_min={min_conf:.2f}",
        ]

        if directional_confidence <= p_neutral + neutral_margin:
            return self._flat_decision(
                raw_alpha, final_signal, confidence, regime_label,
                reasons + ["احتمال neutral از جهت معامله قوی‌تر است — بدون پوزیشن"],
                price, atr, regime_score, trend_strength,
            )

        if directional_confidence < min_conf:
            return self._flat_decision(
                raw_alpha, final_signal, confidence, regime_label,
                reasons + ["اطمینان جهت‌دار زیر آستانه — بدون پوزیشن"],
                price, atr, regime_score, trend_strength,
            )

        if final_signal > min_signal:
            direction = "long"
        elif final_signal < -min_signal:
            direction = "short" if allow_short else "flat"
        else:
            direction = "flat"

        if direction == "flat":
            return self._flat_decision(
                raw_alpha, final_signal, confidence, regime_label,
                reasons + ["سیگنال از dead-zone عبور نکرد"],
                price, atr, regime_score, trend_strength,
            )

        if trend_filter:
            if direction == "long" and trend_strength < -trend_threshold:
                reasons.append(f"هشدار روند: long خلاف روند میانگین‌هاست ({trend_strength:.4f})")
            if direction == "short" and trend_strength > trend_threshold:
                reasons.append(f"هشدار روند: short خلاف روند میانگین‌هاست ({trend_strength:.4f})")

        base_size = min(abs(final_signal), 1.0)
        max_position = float(self._tf_value("max_position_by_tf", self.scfg.get("max_position", 1.0)))

        if self.scfg.get("use_kelly", True):
            p_win = p_up if direction == "long" else p_down
            wlr = self.scfg["take_profit_atr"] / self.scfg["stop_loss_atr"]
            kelly = _kelly_fraction(p_win, wlr) * self.scfg["kelly_fraction"]
            size = base_size * (0.5 + 0.5 * kelly)
            reasons.append(f"کِلی محدود={kelly:.2f}")
        else:
            size = base_size

        position = float(np.clip(size, 0, max_position))
        if direction == "short":
            position = -position

        if direction == "long":
            stop_loss = price - self.scfg["stop_loss_atr"] * atr
            take_profit = price + self.scfg["take_profit_atr"] * atr
        else:
            stop_loss = price + self.scfg["stop_loss_atr"] * atr
            take_profit = price - self.scfg["take_profit_atr"] * atr

        reasons.append(f"پوزیشن={position:+.2f}  SL={stop_loss:.0f}  TP={take_profit:.0f}")

        return {
            "direction": direction,
            "raw_alpha": float(raw_alpha),
            "final_signal": float(final_signal),
            "position": position,
            "confidence": float(directional_confidence),
            "regime": regime_label,
            "regime_score": float(regime_score),
            "stop_loss": float(stop_loss),
            "take_profit": float(take_profit),
            "price": float(price),
            "atr": float(atr),
            "trend_strength": float(trend_strength),
            "reasons": reasons,
        }

    def decide_series(self, proba_df: pd.DataFrame, regime_df: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
        if "atr_pct" in raw.columns:
            atr = raw["atr_pct"] * raw["close"]
        else:
            high, low, close = raw["high"], raw["low"], raw["close"]
            tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
            atr = tr.rolling(14).mean()

        trend_cfg = self.scfg.get("trend_filter", {})
        fast = int(trend_cfg.get("fast_ma", 20))
        slow = int(trend_cfg.get("slow_ma", 100))
        fast_ma = raw["close"].rolling(fast).mean()
        slow_ma = raw["close"].rolling(slow).mean()
        trend_strength = (fast_ma / slow_ma - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        idx = proba_df.index
        rows = []
        for ts in idx:
            d = self.decide_row(
                proba={
                    "p_up": proba_df.at[ts, "p_up"],
                    "p_down": proba_df.at[ts, "p_down"],
                    "p_neutral": proba_df.at[ts, "p_neutral"],
                },
                regime_label=regime_df.at[ts, "regime_label"] if ts in regime_df.index else "neutral",
                regime_score=regime_df.at[ts, "regime_score"] if ts in regime_df.index else 0.0,
                price=float(raw.at[ts, "close"]),
                atr=float(atr.get(ts, np.nan)) if not np.isnan(atr.get(ts, np.nan)) else float(raw.at[ts, "close"]) * 0.02,
                trend_strength=float(trend_strength.get(ts, 0.0)),
            )
            d["timestamp"] = ts
            rows.append(d)

        out = pd.DataFrame(rows).set_index("timestamp")
        out = self._apply_cooldown(out)
        out.attrs["timeframe"] = self.timeframe
        log.info(
            "تصمیم‌ها — long:%d  short:%d  flat:%d",
            (out["direction"] == "long").sum(),
            (out["direction"] == "short").sum(),
            (out["direction"] == "flat").sum(),
        )
        return out

    def _apply_cooldown(self, decisions: pd.DataFrame) -> pd.DataFrame:
        cooldown = int(self._tf_value("cooldown_candles_by_tf", self.scfg.get("cooldown_candles", 0)))
        if cooldown <= 0 or decisions.empty:
            return decisions

        out = decisions.copy()
        last_trade_dir = "flat"
        cooldown_left = 0
        for ts in out.index:
            cur = out.at[ts, "direction"]
            if cooldown_left > 0 and cur != "flat" and cur != last_trade_dir:
                out.at[ts, "direction"] = "flat"
                out.at[ts, "position"] = 0.0
                out.at[ts, "stop_loss"] = None
                out.at[ts, "take_profit"] = None
                reasons = list(out.at[ts, "reasons"]) if isinstance(out.at[ts, "reasons"], list) else []
                reasons.append(f"cooldown فعال ({cooldown_left} کندل باقی‌مانده)")
                out.at[ts, "reasons"] = reasons
                cooldown_left -= 1
                continue

            if cur != last_trade_dir:
                if last_trade_dir != "flat":
                    cooldown_left = cooldown
                last_trade_dir = cur
            elif cooldown_left > 0:
                cooldown_left -= 1
        return out
