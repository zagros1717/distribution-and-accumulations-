"""
Performance metrics. The single most useful number is `fees_pct_of_gross` —
if it's >50% your strategy is paying the exchange to lose for you.
"""
import numpy as np
import pandas as pd


def sharpe(returns: pd.Series, periods_per_year: float = 252 * 10) -> float:
    if returns.std() == 0 or len(returns) < 5:
        return 0.0
    return float(returns.mean() / returns.std() * np.sqrt(periods_per_year))


def sortino(returns: pd.Series, periods_per_year: float = 252 * 10) -> float:
    if len(returns) < 5:
        return 0.0
    downside = returns[returns < 0]
    if len(downside) == 0 or downside.std() == 0:
        return 0.0
    return float(returns.mean() / downside.std() * np.sqrt(periods_per_year))


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    cummax = equity.cummax()
    dd = (equity - cummax) / cummax
    return float(dd.min())


def calmar(equity: pd.Series, total_days: float) -> float:
    if equity.empty or total_days <= 0:
        return 0.0
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (365.0 / total_days) - 1
    mdd = abs(max_drawdown(equity))
    return float(cagr / mdd) if mdd > 0 else 0.0


def summary(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {"n_trades": 0}
    trades = trades.sort_values("entry_ts").reset_index(drop=True)
    starting_equity = 1000.0
    equity = starting_equity + trades["pnl_usd"].cumsum()
    days = (trades["entry_ts"].max() - trades["entry_ts"].min()).total_seconds() / 86400 if len(trades) > 1 else 1
    win = trades["pnl_usd"] > 0
    gross = trades["pnl_usd"] + trades["fees_usd"]
    return {
        "n_trades": int(len(trades)),
        "win_rate": float(win.mean()),
        "net_pnl_usd": float(trades["pnl_usd"].sum()),
        "gross_pnl_usd": float(gross.sum()),
        "fees_usd": float(trades["fees_usd"].sum()),
        "fees_pct_of_gross": float(trades["fees_usd"].sum() / max(abs(gross.sum()), 1) * 100),
        "avg_r_multiple": float(trades["r_multiple"].mean()),
        "median_r_multiple": float(trades["r_multiple"].median()),
        "best_r": float(trades["r_multiple"].max()),
        "worst_r": float(trades["r_multiple"].min()),
        "expectancy_per_trade_usd": float(trades["pnl_usd"].mean()),
        "sharpe": sharpe(trades["r_multiple"]),
        "sortino": sortino(trades["r_multiple"]),
        "max_drawdown_pct": float(max_drawdown(equity) * 100),
        "calmar": calmar(equity, days),
        "trading_days": float(days),
        "trades_per_day": float(len(trades) / max(days, 1)),
        "by_strategy": trades.groupby("strategy")["pnl_usd"].agg(["count", "sum", "mean"]).to_dict() if "strategy" in trades.columns else {},
        "by_close_reason": trades.groupby("reason_close")["pnl_usd"].agg(["count", "sum"]).to_dict() if "reason_close" in trades.columns else {},
    }
