"""
btcalpha.regime.detector
~~~~~~~~~~~~~~~~~~~~~~~~
Causal regime/context detector.

The previous HMM mode fit one Hidden Markov Model on the full dataset and then
assigned states for all rows. That is not causal: the parameters and state path
use future data from validation/test periods. For trading/backtest integrity, the
production detector now defaults to a purely rule-based, rolling/causal score.

Regime output is context and reporting only. Strategy multipliers are currently
set to 1.0 in config, so regime does not block or amplify live signals.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from btcalpha.config import get_config, get_logger

log = get_logger("regime")

_LABELS = {1: "expansion", 0: "neutral", -1: "recession"}


def _rule_based_score(dataset: pd.DataFrame, lookback: int) -> pd.Series:
    """Causal rolling macro/risk-context score in [-1,+1]."""
    idx = dataset.index
    votes = pd.DataFrame(index=idx)

    if "macro_VIX" in dataset:
        vix = dataset["macro_VIX"]
        vix_mean = vix.rolling(lookback, min_periods=lookback).mean()
        vix_std = vix.rolling(lookback, min_periods=lookback).std().replace(0, np.nan)
        vix_z = (vix - vix_mean) / vix_std
        votes["vix"] = (-vix_z).clip(-1, 1)

    if "macro_DXY" in dataset:
        dxy = dataset["macro_DXY"]
        dxy_trend = dxy.pct_change(lookback)
        votes["dxy"] = (-np.sign(dxy_trend) * np.tanh(dxy_trend.abs() * 50)).clip(-1, 1)

    if "macro_US10Y" in dataset:
        y10 = dataset["macro_US10Y"]
        y10_chg = y10.pct_change(lookback)
        votes["us10y"] = np.tanh(y10_chg * 10).clip(-1, 1)

    if "macro_SPX" in dataset:
        spx = dataset["macro_SPX"]
        spx_window = min(200, lookback * 3)
        spx_ma = spx.rolling(spx_window, min_periods=spx_window).mean()
        votes["spx"] = np.tanh((spx / spx_ma - 1) * 8).clip(-1, 1)

    if "macro_GOLD" in dataset and "macro_SPX" in dataset:
        ratio = dataset["macro_GOLD"] / dataset["macro_SPX"]
        ratio_trend = ratio.pct_change(lookback)
        votes["gold_spx"] = (-np.tanh(ratio_trend * 20)).clip(-1, 1)

    if votes.empty:
        log.warning("هیچ ستون ماکرویی برای رژیم قاعده‌محور موجود نیست.")
        return pd.Series(0.0, index=idx)

    return votes.mean(axis=1).fillna(0.0)


def _hmm_score_full_sample_unsafe(dataset: pd.DataFrame, n_states: int) -> Optional[pd.Series]:
    """Legacy full-sample HMM. Kept only for explicit research/debug mode.

    WARNING: This is non-causal and must not be used for production/live/backtest
    metrics. It fits on the full sequence including future validation/test rows.
    """
    try:
        from hmmlearn import hmm  # noqa
    except ImportError:
        log.info("hmmlearn نصب نیست — فقط لایه‌ی قاعده‌محور استفاده می‌شود.")
        return None

    from hmmlearn import hmm

    cols = []
    if "macro_SPX" in dataset:
        cols.append(dataset["macro_SPX"].pct_change())
    if "macro_VIX" in dataset:
        cols.append(dataset["macro_VIX"].pct_change())
    if "close" in dataset:
        cols.append(dataset["close"].pct_change())

    if not cols:
        return None

    feat = pd.concat(cols, axis=1).dropna()
    if len(feat) < 100:
        log.warning("داده برای HMM کافی نیست.")
        return None

    X = feat.values
    try:
        model = hmm.GaussianHMM(
            n_components=n_states, covariance_type="diag", n_iter=200, random_state=42
        )
        model.fit(X)
        hidden = model.predict(X)
    except Exception as exc:  # noqa: BLE001
        log.warning("آموزش HMM ناموفق: %s", exc)
        return None

    state_means = {s: X[hidden == s, 0].mean() for s in range(n_states)}
    ordered = sorted(state_means, key=state_means.get)
    rank_to_score = {ordered[i]: -1 + 2 * i / (n_states - 1) for i in range(n_states)}
    score = pd.Series([rank_to_score[s] for s in hidden], index=feat.index)
    return score.reindex(dataset.index).ffill().fillna(0.0)


def detect_regime(dataset: pd.DataFrame) -> pd.DataFrame:
    """Compute causal regime context for the full dataset.

    Default method is rule_based. Legacy full-sample HMM requires explicitly
    setting regime.allow_full_sample_hmm: true in config; otherwise any method
    containing hmm is downgraded to rule_based.
    """
    cfg = get_config()
    rcfg = cfg["regime"]
    lookback = int(rcfg.get("lookback", 60))

    rule = _rule_based_score(dataset, lookback)
    method = str(rcfg.get("method", "rule_based")).strip().lower()
    allow_hmm = bool(rcfg.get("allow_full_sample_hmm", False))

    score = rule
    source = "rule_based_causal"
    if "hmm" in method:
        if allow_hmm:
            log.warning("استفاده از HMM تمام‌نمونه‌ای فعال است؛ این حالت برای بک‌تست/live causal نیست.")
            hmm_s = _hmm_score_full_sample_unsafe(dataset, int(rcfg.get("hmm_states", 3)))
            if hmm_s is not None:
                score = 0.6 * rule + 0.4 * hmm_s
                source = "rule_plus_full_sample_hmm_unsafe"
        else:
            log.info("HMM غیر causal غیرفعال شد؛ رژیم فقط rule_based causal است.")

    score = score.rolling(5, min_periods=1).mean().clip(-1, 1)

    label_num = pd.Series(0, index=score.index)
    label_num[score > 0.25] = 1
    label_num[score < -0.25] = -1
    label = label_num.map(_LABELS)

    out = pd.DataFrame({"regime_score": score, "regime_label": label}, index=dataset.index)
    out.attrs["regime_source"] = source
    out.attrs["causal"] = source == "rule_based_causal"
    dist = label.value_counts().to_dict()
    log.info("توزیع رژیم: %s | source=%s", dist, source)
    return out
