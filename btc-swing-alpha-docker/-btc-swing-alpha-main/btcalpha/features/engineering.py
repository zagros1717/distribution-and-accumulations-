"""
btcalpha.features.engineering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Feature engineering and label creation for BTC Swing Alpha.

The model no longer has to learn a naive future direction label only. By default
it now uses trade-outcome labeling: what would have been the better executable
trade over the configured horizon after fees, slippage, stop loss and take profit?

Classes remain compatible with the existing strategy/model code:
  0 = short edge
  1 = no trade / neutral
  2 = long edge

Macro data is still loaded into the raw dataset for context/regime display, but
is excluded from model features by default. This keeps VIX/DXY/SPX/etc. from
changing the ML signal directly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from btcalpha.config import get_config, get_logger

log = get_logger("features")


_INTRADAY_MICRO_TFS = {"5m", "15m"}


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    return num / den.replace(0, np.nan)


def _rsi(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(df: pd.DataFrame, window: int) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.rolling(window).mean()


def _macd(close: pd.Series):
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal, macd - signal


def _technical_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    f = pd.DataFrame(index=df.index)
    close = df["close"]

    for k in (1, 3, 5, 10, 20):
        f[f"ret_{k}"] = close.pct_change(k)

    for w in cfg["features"]["ma_windows"]:
        ma = close.rolling(w).mean()
        f[f"ma_ratio_{w}"] = close / ma - 1
        f[f"ma_slope_{w}"] = ma.pct_change(5)

    f[f"rsi_{cfg['features']['rsi_window']}"] = _rsi(
        close, cfg["features"]["rsi_window"]
    )

    macd, _, macd_hist = _macd(close)
    f["macd"] = macd / close
    f["macd_hist"] = macd_hist / close

    atr = _atr(df, cfg["features"]["atr_window"])
    f["atr_pct"] = atr / close
    for w in cfg["features"]["vol_windows"]:
        f[f"vol_{w}"] = close.pct_change().rolling(w).std()

    for w in (14, 30):
        lo = df["low"].rolling(w).min()
        hi = df["high"].rolling(w).max()
        f[f"price_pos_{w}"] = (close - lo) / (hi - lo).replace(0, np.nan)

    return f


def _base_microstructure_features(df: pd.DataFrame) -> pd.DataFrame:
    """Candle-level microstructure proxies used by all timeframes."""
    f = pd.DataFrame(index=df.index)
    o, h, l, c, v = (df["open"], df["high"], df["low"], df["close"], df["volume"])
    rng = (h - l).replace(0, np.nan)

    f["close_loc"] = (c - l) / rng
    f["body_ratio"] = (c - o).abs() / rng
    f["upper_wick"] = (h - np.maximum(o, c)) / rng
    f["lower_wick"] = (np.minimum(o, c) - l) / rng

    for w in (10, 20, 50):
        f[f"vol_rel_{w}"] = v / v.rolling(w).mean()

    direction = np.sign(c.diff()).fillna(0)
    obv = (direction * v).cumsum()
    f["obv_slope"] = obv.diff(10) / v.rolling(10).mean()

    dollar_vol = c * v
    signed_dv = direction * dollar_vol
    f["signed_dv_5"] = signed_dv.rolling(5).sum() / dollar_vol.rolling(5).sum()
    f["signed_dv_20"] = signed_dv.rolling(20).sum() / dollar_vol.rolling(20).sum()

    if "x_spread" in df.columns:
        f["x_spread"] = df["x_spread"].fillna(0.0)
        spread_std = df["x_spread"].rolling(50).std().replace(0, np.nan)
        spread_mean = df["x_spread"].rolling(50).mean()
        f["x_spread_z"] = ((df["x_spread"] - spread_mean) / spread_std).fillna(0.0)
    if "x_count" in df.columns:
        f["x_count"] = df["x_count"].fillna(1)

    return f


def _intraday_microstructure_proxy_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extra order-flow-like features for 5m/15m without raw book/trade storage."""
    f = pd.DataFrame(index=df.index)
    o, h, l, c, v = (df["open"], df["high"], df["low"], df["close"], df["volume"])
    rng = (h - l).replace(0, np.nan)
    ret1 = c.pct_change()
    dollar_vol = (c * v).replace(0, np.nan)

    close_loc = (c - l) / rng
    signed_close_loc = (close_loc * 2.0) - 1.0
    signed_body = _safe_div(c - o, rng)
    upper_wick = _safe_div(h - np.maximum(o, c), rng)
    lower_wick = _safe_div(np.minimum(o, c) - l, rng)
    wick_imbalance = lower_wick - upper_wick

    direction = np.sign(c - o).replace(0, np.nan).fillna(np.sign(c.diff()).fillna(0))
    signed_volume = direction * v
    signed_dollar_volume = direction * dollar_vol
    cvd_proxy = signed_volume.cumsum()

    f["ms_range_pct"] = rng / c
    f["ms_hl_spread_z50"] = (f["ms_range_pct"] - f["ms_range_pct"].rolling(50).mean()) / f["ms_range_pct"].rolling(50).std().replace(0, np.nan)
    f["ms_signed_close_loc"] = signed_close_loc
    f["ms_signed_body"] = signed_body
    f["ms_wick_imbalance"] = wick_imbalance

    for w in (3, 6, 12, 24):
        f[f"ms_volume_pressure_{w}"] = signed_volume.rolling(w).sum() / v.rolling(w).sum()
        f[f"ms_dollar_pressure_{w}"] = signed_dollar_volume.rolling(w).sum() / dollar_vol.rolling(w).sum()

    f["ms_cvd_slope_12"] = cvd_proxy.diff(12) / v.rolling(12).sum()
    f["ms_cvd_slope_24"] = cvd_proxy.diff(24) / v.rolling(24).sum()

    vol_mean_50 = v.rolling(50).mean()
    vol_std_50 = v.rolling(50).std().replace(0, np.nan)
    f["ms_volume_z50"] = (v - vol_mean_50) / vol_std_50
    f["ms_amihud_20"] = (ret1.abs() / dollar_vol).rolling(20).mean()
    f["ms_price_impact_12"] = ret1.rolling(12).sum() / v.rolling(12).sum()

    typical = (h + l + c) / 3.0
    for w in (20, 50):
        vwap = (typical * v).rolling(w).sum() / v.rolling(w).sum()
        f[f"ms_vwap_dist_{w}"] = c / vwap - 1.0

    prev_hi_20 = h.shift(1).rolling(20).max()
    prev_lo_20 = l.shift(1).rolling(20).min()
    sweep_up = ((h > prev_hi_20) & (c < prev_hi_20) & (upper_wick > 0.35)).astype(float)
    sweep_down = ((l < prev_lo_20) & (c > prev_lo_20) & (lower_wick > 0.35)).astype(float)
    f["ms_liquidity_sweep_up"] = sweep_up
    f["ms_liquidity_sweep_down"] = sweep_down
    f["ms_sweep_balance_50"] = sweep_down.rolling(50).sum() - sweep_up.rolling(50).sum()

    for w in (10, 20):
        directional = (c - c.shift(w)).abs()
        total_range = rng.rolling(w).sum()
        f[f"ms_efficiency_{w}"] = directional / total_range

    f["ms_large_pressure"] = np.sign(signed_body).fillna(0) * (
        (f["ms_volume_z50"] > 1.5).astype(float)
        * (f["ms_hl_spread_z50"] > 1.0).astype(float)
        * signed_close_loc.abs().clip(0, 1)
    )

    return f


def _microstructure_features(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    base = _base_microstructure_features(df)
    if timeframe in _INTRADAY_MICRO_TFS:
        extra = _intraday_microstructure_proxy_features(df)
        return pd.concat([base, extra], axis=1)
    return base


def _macro_features(df: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=df.index)
    macro_cols = [c for c in df.columns if c.startswith("macro_")]
    if not macro_cols:
        return f

    for col in macro_cols:
        s = df[col]
        f[f"{col}_chg5"] = s.pct_change(5)
        f[f"{col}_chg20"] = s.pct_change(20)
        f[f"{col}_z"] = (s - s.rolling(60).mean()) / s.rolling(60).std().replace(0, np.nan)

    return f.fillna(0.0)


def _include_macro_features(cfg: dict) -> bool:
    return bool(cfg.get("features", {}).get("include_macro_features", False))


def _tf_config_value(cfg: dict, key: str, timeframe: str, default):
    raw = cfg.get("features", {}).get(key, default)
    if isinstance(raw, dict):
        return raw.get(timeframe, raw.get("default", default))
    return raw


def _feature_audit(feature_cols: list[str], macro_candidate_count: int, include_macro: bool, label_audit: dict) -> dict:
    macro_features = [c for c in feature_cols if c.startswith("macro_")]
    micro_features = [c for c in feature_cols if c.startswith("ms_")]
    return {
        "include_macro_features": include_macro,
        "macro_candidate_features_available": int(macro_candidate_count),
        "macro_features_used": int(len(macro_features)),
        "macro_features_used_names": macro_features,
        "microstructure_features_used": int(len(micro_features)),
        "total_features_used": int(len(feature_cols)),
        "macro_is_context_only": bool(not include_macro and len(macro_features) == 0),
        "label_audit": label_audit,
    }


def _label_threshold(timeframe: str, cfg: dict) -> float:
    return float(_tf_config_value(cfg, "label_threshold_pct", timeframe, 1.5)) / 100.0


def _trade_outcome_min_edge(timeframe: str, cfg: dict) -> float:
    return float(_tf_config_value(cfg, "trade_outcome_min_edge_pct", timeframe, 0.10)) / 100.0


def _trade_outcome_pnl_pct(entry_px: float, exit_px: float, fee: float, direction: str) -> float:
    if not entry_px or entry_px <= 0 or not np.isfinite(entry_px) or not np.isfinite(exit_px):
        return np.nan
    entry_fee = entry_px * fee
    exit_fee = exit_px * fee
    if direction == "long":
        pnl = exit_px - entry_px - entry_fee - exit_fee
    else:
        pnl = entry_px - exit_px - entry_fee - exit_fee
    return pnl / entry_px


def _first_trade_exit(df: pd.DataFrame, start: int, end: int, stop: float, take: float, direction: str, slip: float) -> tuple[float, str]:
    """Conservative intrabar barrier evaluation.

    If both SL and TP are touched in one candle, stop-loss wins. This avoids
    optimistic labeling from unknown intrabar order.
    """
    for j in range(start, end + 1):
        high = float(df["high"].iloc[j])
        low = float(df["low"].iloc[j])
        if direction == "long":
            if low <= stop:
                return stop * (1.0 - slip), "stop_loss"
            if high >= take:
                return take * (1.0 - slip), "take_profit"
        else:
            if high >= stop:
                return stop * (1.0 + slip), "stop_loss"
            if low <= take:
                return take * (1.0 + slip), "take_profit"
    close_exit = float(df["close"].iloc[end])
    if direction == "long":
        return close_exit * (1.0 - slip), "horizon_close"
    return close_exit * (1.0 + slip), "horizon_close"


def _direction_labels(df: pd.DataFrame, timeframe: str, cfg: dict) -> tuple[pd.Series, dict]:
    horizon = int(cfg["features"]["horizons"][timeframe])
    thr = _label_threshold(timeframe, cfg)

    future_ret = df["close"].shift(-horizon) / df["close"] - 1.0
    label = pd.Series(1, index=df.index, dtype="float")
    label[future_ret > thr] = 2
    label[future_ret < -thr] = 0
    label[future_ret.isna()] = np.nan

    audit = {
        "label_mode": "direction",
        "horizon": horizon,
        "direction_threshold_pct": round(thr * 100, 4),
    }
    return label, audit


def _trade_outcome_labels(df: pd.DataFrame, timeframe: str, cfg: dict) -> tuple[pd.Series, dict]:
    horizon = int(cfg["features"]["horizons"][timeframe])
    fee = float(cfg.get("backtest", {}).get("fee_pct", 0.10)) / 100.0
    slip = float(cfg.get("backtest", {}).get("slippage_pct", 0.05)) / 100.0
    stop_atr = float(cfg.get("strategy", {}).get("stop_loss_atr", 2.0))
    take_atr = float(cfg.get("strategy", {}).get("take_profit_atr", 4.5))
    min_edge = _trade_outcome_min_edge(timeframe, cfg)
    atr = _atr(df, int(cfg["features"].get("atr_window", 14)))

    label = pd.Series(np.nan, index=df.index, dtype="float")
    long_pnl = pd.Series(np.nan, index=df.index, dtype="float")
    short_pnl = pd.Series(np.nan, index=df.index, dtype="float")

    n = len(df)
    for i in range(n):
        entry_i = i + 1
        end_i = i + horizon
        if entry_i >= n or end_i >= n:
            continue
        cur_close = float(df["close"].iloc[i])
        cur_atr = float(atr.iloc[i]) if pd.notna(atr.iloc[i]) else np.nan
        if not np.isfinite(cur_close) or not np.isfinite(cur_atr) or cur_atr <= 0:
            continue

        entry_open = float(df["open"].iloc[entry_i])
        long_entry = entry_open * (1.0 + slip)
        short_entry = entry_open * (1.0 - slip)

        long_stop = cur_close - stop_atr * cur_atr
        long_take = cur_close + take_atr * cur_atr
        short_stop = cur_close + stop_atr * cur_atr
        short_take = cur_close - take_atr * cur_atr

        long_exit, _ = _first_trade_exit(df, entry_i, end_i, long_stop, long_take, "long", slip)
        short_exit, _ = _first_trade_exit(df, entry_i, end_i, short_stop, short_take, "short", slip)
        lp = _trade_outcome_pnl_pct(long_entry, long_exit, fee, "long")
        sp = _trade_outcome_pnl_pct(short_entry, short_exit, fee, "short")
        long_pnl.iloc[i] = lp
        short_pnl.iloc[i] = sp

        best = max(lp, sp)
        if best <= min_edge:
            label.iloc[i] = 1
        elif lp > sp:
            label.iloc[i] = 2
        elif sp > lp:
            label.iloc[i] = 0
        else:
            label.iloc[i] = 1

    valid_lp = long_pnl.dropna()
    valid_sp = short_pnl.dropna()
    audit = {
        "label_mode": "trade_outcome",
        "horizon": horizon,
        "fee_pct": round(fee * 100, 4),
        "slippage_pct": round(slip * 100, 4),
        "stop_loss_atr": stop_atr,
        "take_profit_atr": take_atr,
        "min_edge_pct": round(min_edge * 100, 4),
        "avg_long_outcome_pct": round(float(valid_lp.mean() * 100), 4) if len(valid_lp) else None,
        "avg_short_outcome_pct": round(float(valid_sp.mean() * 100), 4) if len(valid_sp) else None,
    }
    return label, audit


def make_labels(df: pd.DataFrame, timeframe: str, cfg: dict) -> tuple[pd.Series, dict]:
    mode = str(cfg.get("features", {}).get("label_mode", "direction")).strip().lower()
    if mode in {"trade_outcome", "trade-outcome", "outcome"}:
        label, audit = _trade_outcome_labels(df, timeframe, cfg)
    else:
        label, audit = _direction_labels(df, timeframe, cfg)

    dist = label.dropna().astype(int).value_counts().sort_index().to_dict()
    audit["class_distribution"] = {int(k): int(v) for k, v in dist.items()}
    audit["valid_labels"] = int(label.notna().sum())
    log.info("label mode %s | %s | class dist=%s", audit.get("label_mode"), timeframe, dist)
    return label, audit


def build_features(dataset: pd.DataFrame, timeframe: str) -> dict:
    cfg = get_config()
    log.info("ساخت فیچر برای %s ...", timeframe)

    tech = _technical_features(dataset, cfg)
    micro = _microstructure_features(dataset, timeframe)
    macro = _macro_features(dataset)
    include_macro = _include_macro_features(cfg)

    parts = [tech, micro]
    if include_macro:
        parts.append(macro)
        log.info("ماکرو به عنوان فیچر مدل فعال است | macro_features=%d", len(macro.columns))
    else:
        log.info("ماکرو فقط context/regime است و از فیچرهای مدل حذف شد | macro_candidates=%d", len(macro.columns))

    X = pd.concat(parts, axis=1)
    X = X.replace([np.inf, -np.inf], np.nan)

    y, label_audit = make_labels(dataset, timeframe, cfg)

    valid_feature_rows = X.dropna().index
    X = X.loc[valid_feature_rows]
    y = y.loc[valid_feature_rows]

    feature_cols = list(X.columns)
    audit = _feature_audit(feature_cols, len(macro.columns), include_macro, label_audit)
    log.info(
        "تعداد فیچر: %d | تعداد ردیف معتبر: %d | macro_used=%d | label=%s",
        len(feature_cols), len(X), audit["macro_features_used"], label_audit.get("label_mode"),
    )

    return {
        "X": X,
        "y": y,
        "feature_cols": feature_cols,
        "feature_audit": audit,
        "label_audit": label_audit,
        "raw": dataset.loc[valid_feature_rows],
        "timeframe": timeframe,
    }
