"""
btcalpha.backtest.engine
~~~~~~~~~~~~~~~~~~~~~~~~
Event-driven backtest engine for BTC Swing Alpha.

Execution rules:
  - Decision on candle t is executed on candle t+1 open.
  - Fees and slippage are applied on both entry and exit.
  - Stop loss / take profit are checked inside each candle using high/low.
  - The entry candle is checked immediately after the fill, so entry-bar gaps are
    not ignored.
  - Default sizing is fixed 1 BTC per trade, not compounded equity sizing.
  - max_holding_bars is counted inclusively from the entry candle. For a 1d
    horizon of 7, entry at open[i+1] can be held through close[i+7], matching the
    trade_outcome label window [i+1 .. i+7].
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Any

import numpy as np
import pandas as pd

from btcalpha.config import get_config, get_logger

log = get_logger("backtest")


@dataclass
class Trade:
    entry_time: str
    exit_time: str
    direction: str
    entry_price: float
    exit_price: float
    size: float
    pnl: float
    pnl_pct: float
    exit_reason: str


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    buyhold_curve: pd.Series
    trades: List[Trade]
    metrics: Dict[str, float]
    decisions: pd.DataFrame

    def summary(self) -> str:
        m = self.metrics
        return "\n".join([
            "=" * 52,
            "  نتیجه‌ی بک‌تست  —  BTC Swing Alpha",
            "=" * 52,
            f"  حالت بک‌تست        : {m.get('sizing_mode', 'fixed_units')}",
            f"  بازده کل استراتژی   : {m['total_return']:+.1f}%",
            f"  بازده Buy & Hold     : {m['buyhold_return']:+.1f}%",
            f"  CAGR                 : {m['cagr']:+.1f}%",
            f"  نسبت شارپ            : {m['sharpe']:.2f}",
            f"  نسبت سورتینو         : {m['sortino']:.2f}",
            f"  حداکثر افت سرمایه    : {m['max_drawdown']:.1f}%",
            f"  تعداد معاملات        : {m['n_trades']}",
            f"  نرخ برد              : {m['win_rate']:.1f}%",
            f"  فاکتور سود           : {m['profit_factor']:.2f}",
            f"  میانگین سود معامله   : {m['avg_trade_pct']:+.2f}%",
            "=" * 52,
        ])


class Backtester:
    def __init__(self):
        self.cfg = get_config()
        self.bcfg = self.cfg["backtest"]

    def _tf_backtest_value(self, key: str, timeframe: str | None, default: Any) -> Any:
        raw = self.bcfg.get(key, default)
        if isinstance(raw, dict):
            if timeframe is not None:
                return raw.get(timeframe, raw.get("default", default))
            return raw.get("default", default)
        return raw

    def _timeframe_from_decisions(self, decisions: pd.DataFrame) -> str | None:
        tf = decisions.attrs.get("timeframe") if hasattr(decisions, "attrs") else None
        return str(tf) if tf else None

    def _max_holding_bars(self, decisions: pd.DataFrame) -> int | None:
        if not bool(self.bcfg.get("use_max_holding_bars", True)):
            return None
        timeframe = self._timeframe_from_decisions(decisions)
        configured = self._tf_backtest_value("max_holding_bars_by_tf", timeframe, None)
        if configured is None:
            if timeframe and timeframe in self.cfg.get("features", {}).get("horizons", {}):
                configured = self.cfg["features"]["horizons"][timeframe]
            else:
                configured = self.bcfg.get("max_holding_bars", None)
        if configured is None:
            return None
        try:
            val = int(configured)
        except (TypeError, ValueError):
            return None
        return val if val > 0 else None

    @staticmethod
    def _barrier_hit(row: pd.Series, direction: str, stop: float | None, take: float | None) -> tuple[str | None, float | None]:
        """Conservative intrabar barrier check. If both hit, stop wins."""
        if direction == "long":
            if stop is not None and row["low"] <= stop:
                return "stop_loss", float(stop)
            if take is not None and row["high"] >= take:
                return "take_profit", float(take)
        elif direction == "short":
            if stop is not None and row["high"] >= stop:
                return "stop_loss", float(stop)
            if take is not None and row["low"] <= take:
                return "take_profit", float(take)
        return None, None

    def run(self, decisions: pd.DataFrame, raw: pd.DataFrame) -> BacktestResult:
        fee = self.bcfg["fee_pct"] / 100.0
        slip = self.bcfg["slippage_pct"] / 100.0
        cap0 = float(self.bcfg["initial_capital"])
        sizing_mode = self.bcfg.get("sizing_mode", "fixed_units")
        fixed_units = float(self.bcfg.get("fixed_trade_units", 1.0))
        max_holding_bars = self._max_holding_bars(decisions)

        idx = decisions.index.intersection(raw.index)
        decisions = decisions.loc[idx]
        raw = raw.loc[idx]

        cash = cap0
        position = 0.0
        entry_price = 0.0
        entry_time = None
        entry_i = None
        entry_fee = 0.0
        direction = "flat"
        cur_stop = None
        cur_tp = None

        equity = []
        trades: List[Trade] = []
        timestamps = list(idx)

        for i, ts in enumerate(timestamps):
            row = raw.loc[ts]
            dec = decisions.loc[ts]
            price_close = float(row["close"])

            mtm = cash + position * price_close if position != 0 else cash
            equity.append((ts, mtm))

            if position != 0:
                hit_reason, exit_px = self._barrier_hit(row, direction, cur_stop, cur_tp)

                if hit_reason is None and max_holding_bars is not None and entry_i is not None:
                    bars_held_inclusive = i - entry_i + 1
                    if bars_held_inclusive >= max_holding_bars:
                        hit_reason, exit_px = "max_holding", price_close

                if hit_reason:
                    cash, trade = self._close(
                        cash, position, direction, entry_price, entry_time,
                        exit_px, ts, fee, slip, hit_reason, entry_fee,
                    )
                    trades.append(trade)
                    position, direction, cur_stop, cur_tp = 0.0, "flat", None, None
                    entry_fee = 0.0
                    entry_i = None

            if i + 1 >= len(timestamps):
                continue
            next_ts = timestamps[i + 1]
            next_row = raw.loc[next_ts]
            next_open = float(next_row["open"])
            target_dir = dec["direction"]

            if position != 0 and target_dir != direction:
                cash, trade = self._close(
                    cash, position, direction, entry_price, entry_time,
                    next_open, next_ts, fee, slip, "signal_flip", entry_fee,
                )
                trades.append(trade)
                position, direction, cur_stop, cur_tp = 0.0, "flat", None, None
                entry_fee = 0.0
                entry_i = None

            if position == 0 and target_dir in ("long", "short"):
                signal_size = abs(float(dec.get("position", 0.0)))
                if signal_size <= 0:
                    continue

                if target_dir == "long":
                    fill = next_open * (1 + slip)
                else:
                    fill = next_open * (1 - slip)

                if sizing_mode == "compounded_equity":
                    notional = cash * signal_size
                    units = notional / fill if fill else 0.0
                else:
                    units = fixed_units
                    notional = units * fill

                if units <= 0 or notional <= 0:
                    continue

                new_entry_fee = notional * fee
                if target_dir == "long" and cash < notional + new_entry_fee:
                    continue

                cash -= new_entry_fee
                if target_dir == "long":
                    position = units
                    cash -= notional
                else:
                    position = -units
                    cash += notional

                direction = target_dir
                entry_price = fill
                entry_time = next_ts
                entry_i = i + 1
                entry_fee = new_entry_fee
                cur_stop = dec["stop_loss"]
                cur_tp = dec["take_profit"]

                # Entry candle must be checked immediately; otherwise gap-through
                # stops/takes on the first bar are ignored and the backtest is optimistic.
                hit_reason, exit_px = self._barrier_hit(next_row, direction, cur_stop, cur_tp)
                if hit_reason:
                    cash, trade = self._close(
                        cash, position, direction, entry_price, entry_time,
                        exit_px, next_ts, fee, slip, f"entry_bar_{hit_reason}", entry_fee,
                    )
                    trades.append(trade)
                    position, direction, cur_stop, cur_tp = 0.0, "flat", None, None
                    entry_fee = 0.0
                    entry_i = None

        if position != 0:
            last_ts = timestamps[-1]
            last_px = float(raw.loc[last_ts, "close"])
            cash, trade = self._close(
                cash, position, direction, entry_price, entry_time,
                last_px, last_ts, fee, slip, "end", entry_fee,
            )
            trades.append(trade)

        equity_curve = pd.Series({ts: v for ts, v in equity}, name="equity").sort_index()
        buyhold_units = fixed_units if sizing_mode != "compounded_equity" else cap0 / raw["close"].iloc[0]
        buyhold = cap0 + buyhold_units * (raw["close"] - raw["close"].iloc[0])
        buyhold.name = "buyhold"

        metrics = self._metrics(equity_curve, buyhold, trades, raw.index)
        metrics["sizing_mode"] = sizing_mode
        metrics["fixed_trade_units"] = fixed_units if sizing_mode != "compounded_equity" else None
        metrics["max_holding_bars"] = max_holding_bars
        metrics["max_holding_counting"] = "inclusive_from_entry_bar"
        metrics["entry_bar_stops_checked"] = True
        log.info(
            "بک‌تست تمام شد — %d معامله، بازده %.1f%% | sizing=%s | max_hold=%s inclusive",
            len(trades), metrics["total_return"], sizing_mode, max_holding_bars,
        )

        return BacktestResult(
            equity_curve=equity_curve,
            buyhold_curve=buyhold,
            trades=trades,
            metrics=metrics,
            decisions=decisions,
        )

    def _close(self, cash, position, direction, entry_price, entry_time,
               exit_price, exit_time, fee, slip, reason, entry_fee=0.0):
        units = abs(position)
        entry_notional = units * entry_price

        if direction == "long":
            fill = float(exit_price) * (1 - slip)
            proceeds = units * fill
            exit_fee = proceeds * fee
            cash += proceeds - exit_fee
            pnl = units * (fill - entry_price) - entry_fee - exit_fee
        else:
            fill = float(exit_price) * (1 + slip)
            buyback = units * fill
            exit_fee = buyback * fee
            cash -= buyback + exit_fee
            pnl = units * (entry_price - fill) - entry_fee - exit_fee

        pnl_pct = (pnl / entry_notional) * 100 if entry_notional else 0.0
        trade = Trade(
            entry_time=str(entry_time),
            exit_time=str(exit_time),
            direction=direction,
            entry_price=round(entry_price, 2),
            exit_price=round(fill, 2),
            size=round(units, 6),
            pnl=round(pnl, 2),
            pnl_pct=round(pnl_pct, 3),
            exit_reason=reason,
        )
        return cash, trade

    def _metrics(self, equity, buyhold, trades, full_index) -> Dict[str, float]:
        if len(equity) < 2:
            return {k: 0.0 for k in (
                "total_return", "buyhold_return", "cagr", "sharpe", "sortino",
                "max_drawdown", "n_trades", "win_rate", "profit_factor",
                "avg_trade_pct")}

        ret = equity.pct_change().dropna()
        total_return = (equity.iloc[-1] / equity.iloc[0] - 1) * 100
        buyhold_return = (buyhold.iloc[-1] / buyhold.iloc[0] - 1) * 100

        tf = full_index.to_series().diff().median()
        periods_per_year = pd.Timedelta(days=365) / tf if tf else 365
        years = len(equity) / periods_per_year
        cagr = ((equity.iloc[-1] / equity.iloc[0]) ** (1 / max(years, 1e-9)) - 1) * 100

        ann = np.sqrt(periods_per_year)
        sharpe = (ret.mean() / ret.std() * ann) if ret.std() > 0 else 0.0
        downside = ret[ret < 0]
        sortino = (ret.mean() / downside.std() * ann) if len(downside) and downside.std() > 0 else 0.0

        running_max = equity.cummax()
        drawdown = (equity / running_max - 1) * 100
        max_dd = drawdown.min()

        n = len(trades)
        if n:
            pnls = np.array([t.pnl for t in trades])
            wins = pnls[pnls > 0]
            losses = pnls[pnls < 0]
            win_rate = len(wins) / n * 100
            gross_win = wins.sum() if len(wins) else 0.0
            gross_loss = abs(losses.sum()) if len(losses) else 0.0
            profit_factor = gross_win / gross_loss if gross_loss > 0 else float("inf")
            avg_trade_pct = float(np.mean([t.pnl_pct for t in trades]))
        else:
            win_rate = profit_factor = avg_trade_pct = 0.0

        return {
            "total_return": round(total_return, 2),
            "buyhold_return": round(buyhold_return, 2),
            "cagr": round(float(cagr), 2),
            "sharpe": round(float(sharpe), 2),
            "sortino": round(float(sortino), 2),
            "max_drawdown": round(float(max_dd), 2),
            "n_trades": n,
            "win_rate": round(win_rate, 1),
            "profit_factor": round(float(profit_factor), 2) if profit_factor != float("inf") else 999.0,
            "avg_trade_pct": round(avg_trade_pct, 3),
        }
