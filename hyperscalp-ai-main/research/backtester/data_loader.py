"""
Load parquet data produced by data_recorder/record_l2.py and resample to bars
the backtester can iterate over.

Output:
    bars: pandas DataFrame indexed by 1m timestamp (UTC), columns:
        [open, high, low, close, volume, vwap, n_trades, buy_vol, sell_vol]
    book_snaps: list of (ts_ms, bids[5], asks[5]) tuples
    liquidations: DataFrame [ts_ms, coin, side, usd]
    funding: DataFrame [ts_ms, coin, funding]
"""
from pathlib import Path
from typing import Dict, Tuple
import pandas as pd
import numpy as np


def load_trades(data_dir: Path, coin: str) -> pd.DataFrame:
    path = data_dir / "trades"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path, filters=[("coin", "==", coin)])
    if df.empty:
        return df
    df["ts"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
    return df.sort_values("ts").reset_index(drop=True)


def trades_to_bars(trades: pd.DataFrame, freq: str = "1min") -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    trades = trades.set_index("ts")
    g = trades.groupby(pd.Grouper(freq=freq))
    bars = pd.DataFrame({
        "open":  g["px"].first(),
        "high":  g["px"].max(),
        "low":   g["px"].min(),
        "close": g["px"].last(),
        "volume": g["sz"].sum(),
        "n_trades": g["px"].count(),
        "buy_vol":  g.apply(lambda x: x.loc[x["is_buy"], "sz"].sum() if not x.empty else 0.0),
        "sell_vol": g.apply(lambda x: x.loc[~x["is_buy"], "sz"].sum() if not x.empty else 0.0),
    })
    bars["vwap"] = (trades["px"] * trades["sz"]).groupby(pd.Grouper(freq=freq)).sum() / bars["volume"].replace(0, np.nan)
    bars = bars.dropna(subset=["close"]).fillna(method="ffill")
    return bars


def load_liquidations(data_dir: Path) -> pd.DataFrame:
    path = data_dir / "liquidations"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if df.empty:
        return df
    df["ts"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
    return df.sort_values("ts").reset_index(drop=True)


def load_funding(data_dir: Path, coin: str) -> pd.DataFrame:
    path = data_dir / "funding"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if df.empty:
        return df
    df = df[df["coin"] == coin].copy()
    df["ts"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
    return df.sort_values("ts").reset_index(drop=True)


def load_book(data_dir: Path, coin: str) -> pd.DataFrame:
    path = data_dir / "book"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path, filters=[("coin", "==", coin)])
    if df.empty:
        return df
    df["ts"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
    return df.sort_values("ts").reset_index(drop=True)


def load_all(data_dir: Path, coins: list) -> Dict[str, dict]:
    out = {}
    for coin in coins:
        trades = load_trades(data_dir, coin)
        bars = trades_to_bars(trades)
        book = load_book(data_dir, coin)
        funding = load_funding(data_dir, coin)
        out[coin] = {"trades": trades, "bars": bars, "book": book, "funding": funding}
    out["_liquidations"] = load_liquidations(data_dir)
    return out
