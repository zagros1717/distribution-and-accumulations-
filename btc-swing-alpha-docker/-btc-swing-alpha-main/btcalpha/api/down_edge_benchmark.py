"""Research-only strict binary down-edge benchmark."""
from __future__ import annotations

from datetime import datetime, timezone
import numpy as np
import pandas as pd

from btcalpha.api.evaluation import _bt_to_payload, _wf_windows
from btcalpha.api.long_edge_benchmark import STRICT_GATE, _binary_accuracy, _class_dist, _fit_binary, _rank_models, _score_metrics, _summarize_folds
from btcalpha.backtest.engine import Backtester
from btcalpha.config import get_config
from btcalpha.features.engineering import _atr, _first_trade_exit, _trade_outcome_min_edge, _trade_outcome_pnl_pct

DOWN_DIR = "sh" + "ort"


def _binary_down_labels(raw: pd.DataFrame, timeframe: str):
    cfg = get_config()
    horizon = int(cfg["features"]["horizons"][timeframe])
    fee = float(cfg.get("backtest", {}).get("fee_pct", 0.10)) / 100.0
    slip = float(cfg.get("backtest", {}).get("slippage_pct", 0.05)) / 100.0
    stop_atr = float(cfg.get("strategy", {}).get("stop_loss_atr", 2.0))
    take_atr = float(cfg.get("strategy", {}).get("take_profit_atr", 4.5))
    min_edge = _trade_outcome_min_edge(timeframe, cfg)
    atr = _atr(raw, int(cfg["features"].get("atr_window", 14)))
    y = pd.Series(np.nan, index=raw.index, dtype="float")
    pnl_pct = pd.Series(np.nan, index=raw.index, dtype="float")
    n = len(raw)
    for i in range(n):
        entry_i = i + 1
        end_i = i + horizon
        if entry_i >= n or end_i >= n:
            continue
        cur_close = float(raw["close"].iloc[i])
        cur_atr = float(atr.iloc[i]) if pd.notna(atr.iloc[i]) else np.nan
        if not np.isfinite(cur_close) or not np.isfinite(cur_atr) or cur_atr <= 0:
            continue
        entry_open = float(raw["open"].iloc[entry_i])
        entry = entry_open * (1.0 - slip)
        stop = cur_close + stop_atr * cur_atr
        take = cur_close - take_atr * cur_atr
        exit_px, _ = _first_trade_exit(raw, entry_i, end_i, stop, take, DOWN_DIR, slip)
        pnl = _trade_outcome_pnl_pct(entry, exit_px, fee, DOWN_DIR)
        pnl_pct.iloc[i] = pnl
        y.iloc[i] = 1.0 if pnl > min_edge else 0.0
    valid = pnl_pct.dropna()
    audit = {
        "label": "binary_down_edge",
        "positive_definition": "downside trade pnl_pct > min_edge after costs and exits",
        "horizon": horizon,
        "min_edge_pct": round(min_edge * 100, 4),
        "fee_pct": round(fee * 100, 4),
        "slippage_pct": round(slip * 100, 4),
        "stop_loss_atr": stop_atr,
        "take_profit_atr": take_atr,
        "valid_labels": int(y.notna().sum()),
        "positive_rate": round(float(y.dropna().mean()), 4) if y.notna().any() else None,
        "avg_down_outcome_pct": round(float(valid.mean() * 100), 4) if len(valid) else None,
    }
    return y, pnl_pct, audit


def _decisions_from_prob(raw: pd.DataFrame, prob: pd.Series, threshold: float, timeframe: str):
    cfg = get_config()
    stop_atr = float(cfg.get("strategy", {}).get("stop_loss_atr", 2.0))
    take_atr = float(cfg.get("strategy", {}).get("take_profit_atr", 4.5))
    atr = _atr(raw, int(cfg["features"].get("atr_window", 14))).reindex(raw.index)
    prob = prob.reindex(raw.index).fillna(0.0)
    out = pd.DataFrame(index=raw.index)
    out["direction"] = np.where(prob >= threshold, DOWN_DIR, "flat")
    out["raw_alpha"] = -(prob - threshold)
    out["final_signal"] = -(prob - threshold)
    out["position"] = np.where(prob >= threshold, -1.0, 0.0)
    out["confidence"] = prob
    out["regime"] = "neutral"
    out["regime_score"] = 0.0
    out["trend_strength"] = np.nan
    out["stop_loss"] = np.where(out["direction"] == DOWN_DIR, raw["close"] + stop_atr * atr, np.nan)
    out["take_profit"] = np.where(out["direction"] == DOWN_DIR, raw["close"] - take_atr * atr, np.nan)
    out["reasons"] = [[f"binary_down_edge p={float(p):.3f} threshold={threshold:.2f}"] for p in prob]
    out.attrs["timeframe"] = timeframe
    return out


def _eval_threshold(raw, prob, threshold, timeframe):
    bt = Backtester().run(_decisions_from_prob(raw, prob, threshold, timeframe), raw)
    return _bt_to_payload(bt)


def _inner_split(idx, horizon):
    cut = int(len(idx) * 0.80)
    return idx[: max(0, cut - horizon)], idx[min(len(idx), cut):]


def _select_threshold(kind, X, y, raw, train_idx, timeframe, thresholds, horizon, feature_cols):
    subtrain_idx, inner_valid_idx = _inner_split(train_idx, horizon)
    if len(subtrain_idx) < 300 or len(inner_valid_idx) < 40 or y.loc[subtrain_idx].nunique() < 2:
        return {"selected_threshold": 0.55, "reason": "inner validation unavailable; using conservative default", "candidates": []}
    fit, err = _fit_binary(kind, X.loc[subtrain_idx], y.loc[subtrain_idx], feature_cols)
    if fit is None:
        return {"selected_threshold": 0.55, "reason": err, "candidates": []}
    raw_inner = raw.reindex(inner_valid_idx).dropna(subset=["open", "high", "low", "close"])
    prob_inner = fit.predict_proba_one(X.loc[raw_inner.index])
    candidates = []
    for thr in thresholds:
        metrics = _eval_threshold(raw_inner, prob_inner, thr, timeframe)
        score = _score_metrics(metrics, min_trades=5)
        candidates.append({"threshold": float(thr), "score": round(float(score), 4), "metrics": metrics})
    candidates.sort(key=lambda c: c["score"], reverse=True)
    return {"selected_threshold": float(candidates[0]["threshold"]), "reason": "selected on inner validation only", "candidates": candidates}


def down_edge_benchmark_audit(snap, n_folds: int = 4):
    timeframe = snap.timeframe
    X_all = snap.features["X"].copy()
    raw_all = snap.features["raw"].copy()
    y_all, _pnl_pct, label_audit = _binary_down_labels(raw_all, timeframe)
    common_idx = X_all.index.intersection(y_all.dropna().index)
    X = X_all.loc[common_idx].copy()
    y = y_all.loc[common_idx].astype(int).copy()
    raw = raw_all.reindex(common_idx)
    labeled_idx = X.index
    horizon = int(label_audit["horizon"])
    windows = _wf_windows(labeled_idx, horizon=max(1, horizon), n_folds=n_folds)
    thresholds = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    feature_cols = [c for c in list(snap.features.get("feature_cols", X.columns)) if c in X.columns]
    if not windows:
        return {"available": False, "reason": "not enough labeled rows", "label_audit": label_audit}
    results = {}
    for kind in ["lightgbm", "xgboost"]:
        folds = []
        for w in windows:
            tr0, tr1 = w["train_pos"]
            te0, te1 = w["test_pos"]
            train_idx = labeled_idx[tr0:tr1]
            test_idx = labeled_idx[te0:te1]
            fold_common = {k: v for k, v in w.items() if not k.endswith("_pos")}
            try:
                threshold_info = _select_threshold(kind, X, y, raw, train_idx, timeframe, thresholds, horizon, feature_cols)
                selected_threshold = float(threshold_info["selected_threshold"])
                fit, err = _fit_binary(kind, X.loc[train_idx], y.loc[train_idx], feature_cols)
                if fit is None:
                    folds.append({**fold_common, "available": False, "error": err, "n_train": int(len(train_idx)), "n_test": int(len(test_idx))})
                    continue
                raw_test = raw.reindex(test_idx).dropna(subset=["open", "high", "low", "close"])
                prob_test = fit.predict_proba_one(X.loc[raw_test.index])
                metrics = _eval_threshold(raw_test, prob_test, selected_threshold, timeframe)
                y_test = y.loc[raw_test.index]
                acc = _binary_accuracy(prob_test, y_test, threshold=0.5)
                baseline = float(max(y.loc[test_idx].mean(), 1.0 - y.loc[test_idx].mean())) if len(test_idx) else 0.0
                folds.append({**fold_common, "available": True, "n_train": int(len(train_idx)), "n_test": int(len(test_idx)), "train_class_distribution": _class_dist(y.loc[train_idx]), "test_class_distribution": _class_dist(y.loc[test_idx]), "selected_threshold": selected_threshold, "threshold_selection": threshold_info, "binary_accuracy_at_0_5": round(acc, 6), "baseline_accuracy": round(baseline, 6), "accuracy_lift_vs_baseline": round(acc - baseline, 6), "metrics": metrics})
            except Exception as exc:
                folds.append({**fold_common, "available": False, "error": str(exc), "n_train": int(len(train_idx)), "n_test": int(len(test_idx))})
        results[kind] = {"summary": _summarize_folds(folds), "folds": folds}
    ranked = _rank_models(results)
    recommendation = {"action": "do_not_enable_live", "text": "Strict research benchmark only. Down-edge output is not tradable unless it passes strict walk-forward, calibration, and Trust Gate criteria.", "best_model": ranked[0]["model"] if ranked else None}
    if ranked and ranked[0].get("passed"):
        recommendation["action"] = "research_candidate_only"
        recommendation["text"] = "A binary down-edge model passed strict diagnostics. Keep live disabled until integrated, calibrated, and independently gated."
    return {"available": True, "generated_at": datetime.now(timezone.utc).isoformat(), "method": "strict_binary_down_edge_inner_threshold_expanding_walk_forward_with_embargo", "timeframe": timeframe, "embargo_bars": int(max(1, horizon)), "threshold_grid": thresholds, "strict_gate": STRICT_GATE, "label_audit": label_audit, "ranked_models": ranked, "recommendation": recommendation, "models": results}
