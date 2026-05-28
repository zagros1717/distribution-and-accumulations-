"""
Python ports of base44/functions/botExecutor/lib/features.ts.
KEEP THESE IN SYNC with the executor — divergence here is the most common backtest-vs-live discrepancy.
"""
import numpy as np
import pandas as pd


def ema(x: np.ndarray, period: int) -> np.ndarray:
    if len(x) == 0:
        return x
    k = 2 / (period + 1)
    out = np.empty_like(x, dtype=float)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = x[i] * k + out[i - 1] * (1 - k)
    return out


def atr(bars: pd.DataFrame, period: int = 14) -> float:
    if len(bars) < period + 1:
        return 0.0
    h = bars["high"].values
    l = bars["low"].values
    c = bars["close"].values
    tr = np.maximum.reduce([
        h[1:] - l[1:],
        np.abs(h[1:] - c[:-1]),
        np.abs(l[1:] - c[:-1]),
    ])
    return float(np.mean(tr[-period:]))


def adx(bars: pd.DataFrame, period: int = 14) -> float:
    if len(bars) < period * 2:
        return 0.0
    h = bars["high"].values
    l = bars["low"].values
    c = bars["close"].values
    up = h[1:] - h[:-1]
    dn = l[:-1] - l[1:]
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = np.maximum.reduce([h[1:] - l[1:], np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])])

    def wsmooth(arr, p):
        out = np.zeros(len(arr) - p + 1)
        out[0] = np.sum(arr[:p])
        for i in range(p, len(arr)):
            out[i - p + 1] = out[i - p] - out[i - p] / p + arr[i]
        return out

    tr_s = wsmooth(tr, period)
    pdm_s = wsmooth(plus_dm, period)
    mdm_s = wsmooth(minus_dm, period)
    if len(tr_s) == 0:
        return 0.0
    pdi = np.where(tr_s == 0, 0.0, 100 * pdm_s / tr_s)
    mdi = np.where(tr_s == 0, 0.0, 100 * mdm_s / tr_s)
    dx = np.where((pdi + mdi) == 0, 0.0, 100 * np.abs(pdi - mdi) / (pdi + mdi))
    if len(dx) < period:
        return 0.0
    return float(np.mean(dx[-period:]))


def choppiness(bars: pd.DataFrame, period: int = 14) -> float:
    if len(bars) < period + 1:
        return 0.0
    recent = bars.iloc[-period:]
    h = recent["high"].values; l = recent["low"].values; c = recent["close"].values
    pc = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum.reduce([h - l, np.abs(h - pc), np.abs(l - pc)])
    tr_sum = float(np.sum(tr))
    hi = float(np.max(h)); lo = float(np.min(l))
    if tr_sum == 0 or hi - lo == 0:
        return 0.0
    return 100 * np.log10(tr_sum / (hi - lo)) / np.log10(period)


def vwap_bands(bars: pd.DataFrame):
    if bars.empty:
        return 0.0, 0.0, 0.0
    tp = (bars["high"] + bars["low"] + bars["close"]) / 3
    pv = (tp * bars["volume"]).sum()
    v = bars["volume"].sum()
    if v <= 0:
        return float(bars["close"].iloc[-1]), 0.0, 0.0
    vwap = pv / v
    var = (bars["volume"] * (tp - vwap) ** 2).sum() / v
    sigma = float(np.sqrt(var))
    last_close = float(bars["close"].iloc[-1])
    deviation = (last_close - vwap) / sigma if sigma > 0 else 0.0
    return float(vwap), sigma, deviation


def ema_stack(bars: pd.DataFrame):
    if len(bars) < 100:
        return 0
    closes = bars["close"].values
    e20 = ema(closes, 20); e50 = ema(closes, 50); e100 = ema(closes, 100)
    if e20[-1] > e50[-1] > e100[-1]:
        return 1
    if e20[-1] < e50[-1] < e100[-1]:
        return -1
    return 0


def cvd_series(trades: pd.DataFrame, end_ts: pd.Timestamp, window_ms: int, buckets: int = 12):
    if trades.empty:
        return [], []
    start = end_ts - pd.Timedelta(milliseconds=window_ms)
    sub = trades[(trades["ts"] >= start) & (trades["ts"] <= end_ts)]
    if len(sub) < buckets:
        return [], []
    sub = sub.copy()
    sub["bucket"] = pd.cut(sub["ts"], bins=buckets, labels=False)
    cvd_list = []; price_list = []
    cum = 0.0
    for b in range(buckets):
        bsub = sub[sub["bucket"] == b]
        delta = bsub.loc[bsub["is_buy"], "sz"].sum() - bsub.loc[~bsub["is_buy"], "sz"].sum()
        cum += float(delta)
        cvd_list.append(cum)
        price_list.append(float(bsub["px"].iloc[-1]) if not bsub.empty else (price_list[-1] if price_list else 0.0))
    return cvd_list, price_list


def cvd_divergence(trades: pd.DataFrame, end_ts: pd.Timestamp, window_ms: int = 8 * 60 * 1000):
    cvd, price = cvd_series(trades, end_ts, window_ms)
    if len(cvd) < 6:
        return 0, 0.0
    half = len(cvd) // 2
    p_hi1, p_hi2 = max(price[:half]), max(price[half:])
    p_lo1, p_lo2 = min(price[:half]), min(price[half:])
    c_hi1, c_hi2 = max(cvd[:half]), max(cvd[half:])
    c_lo1, c_lo2 = min(cvd[:half]), min(cvd[half:])
    if p_hi2 > p_hi1 and c_hi2 < c_hi1:
        return -1, 1.0
    if p_lo2 < p_lo1 and c_lo2 > c_lo1:
        return 1, 1.0
    return 0, 0.0
