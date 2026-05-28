"""
btcalpha.api.evaluation
~~~~~~~~~~~~~~~~~~~~~~~
Out-of-sample evaluation helpers for API responses.

This module intentionally separates:
  - full long+short strategy result
  - long-only diagnostic result
  - short-only diagnostic result
  - model-quality audit
  - walk-forward audit with an embargo gap
  - threshold/risk sweep audit using valid -> test discipline

Walk-forward and threshold-sweep audits are diagnostic only. They never overwrite
production model files or config. Threshold sweep selects parameters only on the
validation split and then reports the selected set on the test split.
"""
from __future__ import annotations

import copy
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict

import numpy as np
import pandas as pd

from btcalpha.backtest.engine import Backtester, BacktestResult
from btcalpha.live.engine import PipelineSnapshot
from btcalpha.model.alpha_model import AlphaModel
from btcalpha.strategy.signal import Strategy


def _trades_summary(trades, equity_curve=None) -> dict:
    pnls = [float(t.pnl) for t in trades]
    pct = [float(t.pnl_pct) for t in trades]
    win_pnls = [p for p in pnls if p > 0]
    loss_pnls = [p for p in pnls if p < 0]
    gross_profit = sum(win_pnls)
    gross_loss = abs(sum(loss_pnls))
    net_pnl = sum(pnls)
    n = len(trades)
    wins = len(win_pnls)
    losses = len(loss_pnls)
    long_pnl = sum(float(t.pnl) for t in trades if t.direction == "long")
    short_pnl = sum(float(t.pnl) for t in trades if t.direction == "short")
    long_count = sum(1 for t in trades if t.direction == "long")
    short_count = sum(1 for t in trades if t.direction == "short")
    final_equity = float(equity_curve.iloc[-1]) if equity_curve is not None and len(equity_curve) else None
    initial_equity = float(equity_curve.iloc[0]) if equity_curve is not None and len(equity_curve) else None

    return {
        "n_trades": n,
        "wins": wins,
        "losses": losses,
        "flats": n - wins - losses,
        "gross_profit": round(gross_profit, 2),
        "gross_loss": round(gross_loss, 2),
        "net_pnl": round(net_pnl, 2),
        "avg_win": round(gross_profit / wins, 2) if wins else 0.0,
        "avg_loss": round(-gross_loss / losses, 2) if losses else 0.0,
        "avg_trade_pnl": round(net_pnl / n, 2) if n else 0.0,
        "avg_trade_pct": round(sum(pct) / n, 3) if n else 0.0,
        "best_trade": round(max(pnls), 2) if pnls else 0.0,
        "worst_trade": round(min(pnls), 2) if pnls else 0.0,
        "win_rate": round((wins / n * 100), 1) if n else 0.0,
        "profit_factor": round(gross_profit / gross_loss, 2) if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0),
        "long_count": long_count,
        "short_count": short_count,
        "long_pnl": round(long_pnl, 2),
        "short_pnl": round(short_pnl, 2),
        "initial_equity": round(initial_equity, 2) if initial_equity is not None else None,
        "final_equity": round(final_equity, 2) if final_equity is not None else None,
        "equity_pnl": round(final_equity - initial_equity, 2) if final_equity is not None and initial_equity is not None else None,
    }


def _bt_to_payload(bt: BacktestResult) -> dict:
    summary = _trades_summary(bt.trades, bt.equity_curve)
    return {**bt.metrics, "trade_summary": summary}


def _empty_segment(reason: str) -> dict:
    return {
        "available": False,
        "reason": reason,
        "metrics": {},
        "trade_summary": {},
        "first_timestamp": None,
        "last_timestamp": None,
        "n_rows": 0,
    }


def _filter_decisions(decisions: pd.DataFrame, mode: str) -> pd.DataFrame:
    out = decisions.copy(deep=True)
    if mode == "long_short":
        return out
    if mode == "long_only":
        mask = out["direction"] == "short"
    elif mode == "short_only":
        mask = out["direction"] == "long"
    else:
        raise ValueError(f"unknown side-audit mode: {mode}")

    out.loc[mask, "direction"] = "flat"
    out.loc[mask, "position"] = 0.0
    out.loc[mask, "stop_loss"] = None
    out.loc[mask, "take_profit"] = None
    if "reasons" in out.columns:
        for ts in out.index[mask]:
            reasons = out.at[ts, "reasons"] if isinstance(out.at[ts, "reasons"], list) else []
            out.at[ts, "reasons"] = list(reasons) + [f"side_audit:{mode} filtered this side"]
    return out


def _run_segment(name: str, idx, snap: PipelineSnapshot, mode: str = "long_short") -> dict:
    idx = idx.intersection(snap.decisions.index).intersection(snap.features["raw"].index)
    if len(idx) < 5:
        return _empty_segment(f"not enough rows for {name}")

    decisions = _filter_decisions(snap.decisions.loc[idx], mode)
    raw = snap.features["raw"].loc[idx]
    bt = Backtester().run(decisions, raw)
    return {
        "available": True,
        "metrics": _bt_to_payload(bt),
        "first_timestamp": str(idx[0]),
        "last_timestamp": str(idx[-1]),
        "n_rows": int(len(idx)),
    }


def _split_indices(snap: PipelineSnapshot) -> dict:
    meta = snap.model.meta
    if meta is None:
        return {}
    y = snap.features["y"]
    labeled_idx = y[y.notna()].index
    n_train = int(meta.n_train)
    n_valid = int(meta.n_valid)
    n_test = int(meta.n_test)
    return {
        "train": labeled_idx[:n_train],
        "valid": labeled_idx[n_train:n_train + n_valid],
        "test": labeled_idx[n_train + n_valid:n_train + n_valid + n_test],
        "split": {
            "n_train": n_train,
            "n_valid": n_valid,
            "n_test": n_test,
            "labeled_rows": int(len(labeled_idx)),
        },
    }


def backtest_segments(snap: PipelineSnapshot) -> Dict[str, dict]:
    if snap.model.meta is None:
        return {
            "train": _empty_segment("model metadata missing"),
            "valid": _empty_segment("model metadata missing"),
            "test": _empty_segment("model metadata missing"),
        }

    idx = _split_indices(snap)
    return {
        "train": _run_segment("train", idx["train"], snap),
        "valid": _run_segment("valid", idx["valid"], snap),
        "test": _run_segment("test", idx["test"], snap),
        "split": idx["split"],
    }


def side_audit(snap: PipelineSnapshot) -> Dict[str, dict]:
    if snap.model.meta is None:
        return {"available": False, "reason": "model metadata missing"}

    idx = _split_indices(snap)
    modes = ("long_short", "long_only", "short_only")
    out: Dict[str, dict] = {"available": True, "split": idx["split"], "modes": {}}
    for mode in modes:
        out["modes"][mode] = {
            "train": _run_segment(f"train:{mode}", idx["train"], snap, mode=mode),
            "valid": _run_segment(f"valid:{mode}", idx["valid"], snap, mode=mode),
            "test": _run_segment(f"test:{mode}", idx["test"], snap, mode=mode),
        }

    test_summary = {}
    for mode in modes:
        seg = out["modes"][mode]["test"]
        metrics = seg.get("metrics", {}) if seg.get("available") else {}
        ts = metrics.get("trade_summary", {})
        test_summary[mode] = {
            "total_return": metrics.get("total_return"),
            "profit_factor": metrics.get("profit_factor"),
            "sharpe": metrics.get("sharpe"),
            "max_drawdown": metrics.get("max_drawdown"),
            "n_trades": metrics.get("n_trades"),
            "net_pnl": ts.get("net_pnl"),
            "long_pnl": ts.get("long_pnl"),
            "short_pnl": ts.get("short_pnl"),
        }
    out["test_summary"] = test_summary
    return out


def model_quality_audit(snap: PipelineSnapshot) -> Dict[str, Any]:
    meta = snap.model.meta
    if meta is None:
        return {"available": False, "reason": "model metadata missing"}

    lift = float(getattr(meta, "accuracy_lift_vs_baseline", 0.0) or 0.0)
    valid_lift = float((getattr(meta, "valid_accuracy", 0.0) or 0.0) - (getattr(meta, "baseline_valid_accuracy", 0.0) or 0.0))
    test_lift = float((getattr(meta, "test_accuracy", 0.0) or 0.0) - (getattr(meta, "baseline_test_accuracy", 0.0) or 0.0))
    warnings = []
    if valid_lift < 0:
        warnings.append(f"valid accuracy is below baseline by {abs(valid_lift) * 100:.2f} percentage points")
    elif valid_lift < 0.01:
        warnings.append(f"valid accuracy lift is weak: {valid_lift * 100:.2f} percentage points")
    if test_lift < 0:
        warnings.append(f"test accuracy is below baseline by {abs(test_lift) * 100:.2f} percentage points")
    elif test_lift < 0.01:
        warnings.append(f"test accuracy lift is weak: {test_lift * 100:.2f} percentage points")

    return {
        "available": True,
        "valid_accuracy": getattr(meta, "valid_accuracy", None),
        "valid_baseline_accuracy": getattr(meta, "baseline_valid_accuracy", None),
        "valid_lift_vs_baseline": round(valid_lift, 6),
        "test_accuracy": getattr(meta, "test_accuracy", None),
        "test_baseline_accuracy": getattr(meta, "baseline_test_accuracy", None),
        "test_lift_vs_baseline": round(test_lift, 6),
        "reported_lift_vs_baseline": round(lift, 6),
        "majority_class": getattr(meta, "majority_class", None),
        "warnings": warnings,
        "passes_basic_model_edge": valid_lift > 0 and test_lift > 0,
    }


def _wf_windows(labeled_idx: pd.Index, horizon: int, n_folds: int) -> list[dict]:
    n = len(labeled_idx)
    if n < 600:
        return []
    folds = max(2, min(int(n_folds), 8))
    min_train = max(500, int(n * 0.45))
    remaining = n - min_train - horizon
    if remaining < folds * 40:
        folds = max(2, remaining // 40)
    if folds < 2:
        return []
    test_size = max(40, remaining // folds)
    windows = []
    for i in range(folds):
        train_start = 0
        train_end = min_train + i * test_size
        test_start = train_end + horizon
        test_end = min(test_start + test_size, n)
        if test_end - test_start < 20:
            continue
        windows.append({
            "fold": i + 1,
            "train_pos": [train_start, train_end],
            "embargo_bars": int(horizon),
            "test_pos": [test_start, test_end],
            "train_start": str(labeled_idx[train_start]),
            "train_end": str(labeled_idx[train_end - 1]),
            "test_start": str(labeled_idx[test_start]),
            "test_end": str(labeled_idx[test_end - 1]),
        })
    return windows


def _fit_temp_model(timeframe: str, Xtr: pd.DataFrame, ytr: pd.Series) -> AlphaModel:
    m = AlphaModel(timeframe)
    m.feature_cols = list(Xtr.columns)
    m._fit_scaler(Xtr)
    Xtr_s = m._scale(Xtr)
    m.model = m._build_estimator()

    classes = np.unique(ytr.astype(int))
    if len(classes) < 2:
        raise ValueError(f"not enough class variety in walk-forward train: {classes.tolist()}")

    sample_weight = m._sample_weight(ytr.astype(int))
    if sample_weight is None:
        m.model.fit(Xtr_s, ytr.astype(int))
    else:
        try:
            m.model.fit(Xtr_s, ytr.astype(int), sample_weight=sample_weight)
        except TypeError:
            m.model.fit(Xtr_s, ytr.astype(int))
    return m


def _baseline_accuracy(ytr: pd.Series, yte: pd.Series) -> tuple[int, float]:
    majority = int(ytr.astype(int).value_counts().idxmax())
    acc = float((yte.astype(int) == majority).mean()) if len(yte) else 0.0
    return majority, acc


def walk_forward_audit(snap: PipelineSnapshot, n_folds: int = 4) -> Dict[str, Any]:
    feat = snap.features
    X_all = feat["X"]
    y_all = feat["y"]
    mask = y_all.notna()
    X = X_all.loc[mask].copy()
    y = y_all.loc[mask].astype(int).copy()
    labeled_idx = X.index
    horizon = int(snap.model.meta.horizon if snap.model.meta else 0)
    windows = _wf_windows(labeled_idx, horizon=max(1, horizon), n_folds=n_folds)
    if not windows:
        return {"available": False, "reason": "not enough labeled rows for walk-forward audit"}

    folds = []
    for w in windows:
        tr0, tr1 = w["train_pos"]
        te0, te1 = w["test_pos"]
        train_idx = labeled_idx[tr0:tr1]
        test_idx = labeled_idx[te0:te1]
        Xtr, ytr = X.loc[train_idx], y.loc[train_idx]
        Xte, yte = X.loc[test_idx], y.loc[test_idx]
        try:
            temp_model = _fit_temp_model(snap.model.timeframe, Xtr, ytr)
            proba = temp_model.predict_proba(Xte)
            pred = proba[["p_down", "p_neutral", "p_up"]].to_numpy().argmax(axis=1)
            acc = float((pred == yte.to_numpy()).mean()) if len(yte) else 0.0
            majority, base_acc = _baseline_accuracy(ytr, yte)
            lift = acc - base_acc

            regime = snap.regime.reindex(test_idx).ffill().bfill()
            raw = snap.features["raw"].reindex(test_idx).dropna(subset=["open", "high", "low", "close"])
            proba = proba.reindex(raw.index)
            regime = regime.reindex(raw.index).ffill().bfill()
            decisions = Strategy(snap.model.timeframe).decide_series(proba, regime, raw)
            bt = Backtester().run(decisions, raw)
            metrics = _bt_to_payload(bt)

            folds.append({
                **{k: v for k, v in w.items() if not k.endswith("_pos")},
                "available": True,
                "n_train": int(len(Xtr)),
                "n_test": int(len(Xte)),
                "train_class_distribution": dict(Counter(ytr.astype(int).tolist())),
                "test_class_distribution": dict(Counter(yte.astype(int).tolist())),
                "majority_class": majority,
                "accuracy": round(acc, 6),
                "baseline_accuracy": round(base_acc, 6),
                "lift_vs_baseline": round(lift, 6),
                "class_weighting": temp_model._class_weighting_mode(),
                "metrics": metrics,
            })
        except Exception as exc:  # noqa: BLE001
            folds.append({
                **{k: v for k, v in w.items() if not k.endswith("_pos")},
                "available": False,
                "error": str(exc),
                "n_train": int(len(Xtr)),
                "n_test": int(len(Xte)),
            })

    ok = [f for f in folds if f.get("available")]
    positive_return = [f for f in ok if float(f["metrics"].get("total_return", 0.0) or 0.0) > 0]
    positive_pf = [f for f in ok if float(f["metrics"].get("profit_factor", 0.0) or 0.0) >= 1.10]
    positive_lift = [f for f in ok if float(f.get("lift_vs_baseline", 0.0) or 0.0) > 0]
    returns = [float(f["metrics"].get("total_return", 0.0) or 0.0) for f in ok]
    pfs = [float(f["metrics"].get("profit_factor", 0.0) or 0.0) for f in ok]
    sharpes = [float(f["metrics"].get("sharpe", 0.0) or 0.0) for f in ok]
    lifts = [float(f.get("lift_vs_baseline", 0.0) or 0.0) for f in ok]

    passed = bool(ok) and len(positive_return) >= max(1, int(np.ceil(len(ok) * 0.6))) and len(positive_pf) >= max(1, int(np.ceil(len(ok) * 0.6)))
    warnings = []
    if ok and len(positive_lift) < max(1, int(np.ceil(len(ok) * 0.5))):
        warnings.append("model accuracy lift is not positive in most walk-forward folds")
    if ok and np.nanmean(returns) <= 0:
        warnings.append("average walk-forward return is not positive")
    if ok and np.nanmean(pfs) < 1.10:
        warnings.append("average walk-forward profit factor is below 1.10")

    return {
        "available": bool(ok),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "method": "expanding_walk_forward_with_embargo",
        "n_requested_folds": int(n_folds),
        "n_available_folds": int(len(ok)),
        "embargo_bars": int(max(1, horizon)),
        "class_weighting": ok[0].get("class_weighting") if ok else None,
        "passed": passed,
        "summary": {
            "positive_return_folds": len(positive_return),
            "positive_profit_factor_folds": len(positive_pf),
            "positive_model_lift_folds": len(positive_lift),
            "avg_return": round(float(np.nanmean(returns)), 3) if returns else None,
            "avg_profit_factor": round(float(np.nanmean(pfs)), 3) if pfs else None,
            "avg_sharpe": round(float(np.nanmean(sharpes)), 3) if sharpes else None,
            "avg_lift_vs_baseline": round(float(np.nanmean(lifts)), 6) if lifts else None,
        },
        "warnings": warnings,
        "folds": folds,
    }


def _strategy_with_overrides(timeframe: str, params: dict) -> Strategy:
    s = Strategy(timeframe)
    s.scfg = copy.deepcopy(s.scfg)
    for key, value in (
        ("min_confidence_by_tf", params["min_confidence"]),
        ("min_abs_signal_by_tf", params["min_abs_signal"]),
        ("neutral_margin_by_tf", params["neutral_margin"]),
    ):
        if not isinstance(s.scfg.get(key), dict):
            s.scfg[key] = {}
        s.scfg[key][timeframe] = float(value)
    return s


def _eval_params_on_index(snap: PipelineSnapshot, idx: pd.Index, params: dict) -> dict:
    idx = idx.intersection(snap.features["X"].index).intersection(snap.features["raw"].index)
    if len(idx) < 20:
        return _empty_segment("not enough rows for parameter evaluation")
    raw = snap.features["raw"].loc[idx].dropna(subset=["open", "high", "low", "close"])
    if len(raw) < 20:
        return _empty_segment("not enough raw rows for parameter evaluation")
    X = snap.features["X"].loc[raw.index]
    proba = snap.model.predict_proba(X)
    regime = snap.regime.reindex(raw.index).ffill().bfill()
    decisions = _strategy_with_overrides(snap.model.timeframe, params).decide_series(proba, regime, raw)
    bt = Backtester().run(decisions, raw)
    return {
        "available": True,
        "params": dict(params),
        "metrics": _bt_to_payload(bt),
        "first_timestamp": str(raw.index[0]),
        "last_timestamp": str(raw.index[-1]),
        "n_rows": int(len(raw)),
    }


def _candidate_score(metrics: dict, min_trades: int = 20) -> float:
    n_trades = int(metrics.get("n_trades", 0) or 0)
    if n_trades < min_trades:
        return -1e9
    total_return = float(metrics.get("total_return", 0.0) or 0.0)
    pf = float(metrics.get("profit_factor", 0.0) or 0.0)
    sharpe = float(metrics.get("sharpe", 0.0) or 0.0)
    max_dd = float(metrics.get("max_drawdown", 0.0) or 0.0)
    # Conservative objective: reward return, PF and Sharpe; penalize drawdown.
    return total_return + 12.0 * (pf - 1.0) + 5.0 * sharpe + 0.25 * max_dd


def threshold_sweep_audit(snap: PipelineSnapshot, top_n: int = 12) -> Dict[str, Any]:
    if snap.model.meta is None:
        return {"available": False, "reason": "model metadata missing"}

    idx = _split_indices(snap)
    valid_idx = idx["valid"]
    test_idx = idx["test"]
    timeframe = snap.model.timeframe

    conf_grid = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65]
    signal_grid = [0.05, 0.065, 0.08, 0.10, 0.12, 0.15]
    neutral_grid = [0.00, 0.02, 0.04, 0.06, 0.08]

    candidates = []
    for min_conf in conf_grid:
        for min_signal in signal_grid:
            for neutral_margin in neutral_grid:
                params = {
                    "min_confidence": min_conf,
                    "min_abs_signal": min_signal,
                    "neutral_margin": neutral_margin,
                }
                valid_result = _eval_params_on_index(snap, valid_idx, params)
                if not valid_result.get("available"):
                    continue
                vm = valid_result["metrics"]
                score = _candidate_score(vm, min_trades=15)
                candidates.append({
                    "params": params,
                    "valid_score": round(float(score), 4),
                    "valid_metrics": vm,
                })

    if not candidates:
        return {"available": False, "reason": "no valid candidates produced trades"}

    candidates.sort(key=lambda x: x["valid_score"], reverse=True)
    top = candidates[:max(1, min(int(top_n), 25))]
    selected = top[0]
    test_result = _eval_params_on_index(snap, test_idx, selected["params"])

    current_params = {
        "min_confidence": float(Strategy(timeframe)._tf_value("min_confidence_by_tf", 0.45)),
        "min_abs_signal": float(Strategy(timeframe)._tf_value("min_abs_signal_by_tf", 0.05)),
        "neutral_margin": float(Strategy(timeframe)._tf_value("neutral_margin_by_tf", 0.02)),
    }
    current_valid = _eval_params_on_index(snap, valid_idx, current_params)
    current_test = _eval_params_on_index(snap, test_idx, current_params)

    tm = test_result.get("metrics", {}) if test_result.get("available") else {}
    passed = bool(test_result.get("available")) and (
        float(tm.get("total_return", 0.0) or 0.0) > 0
        and float(tm.get("profit_factor", 0.0) or 0.0) >= 1.10
        and float(tm.get("max_drawdown", 0.0) or 0.0) >= -25.0
        and int(tm.get("n_trades", 0) or 0) >= 5
    )

    warnings = []
    if not passed:
        if float(tm.get("total_return", 0.0) or 0.0) <= 0:
            warnings.append("selected valid-tuned params are not profitable on test")
        if float(tm.get("profit_factor", 0.0) or 0.0) < 1.10:
            warnings.append("selected valid-tuned params have test profit factor below 1.10")
        if float(tm.get("max_drawdown", 0.0) or 0.0) < -25.0:
            warnings.append("selected valid-tuned params have test drawdown deeper than -25%")

    return {
        "available": True,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "method": "valid_grid_search_then_test_once",
        "timeframe": timeframe,
        "grid": {
            "min_confidence": conf_grid,
            "min_abs_signal": signal_grid,
            "neutral_margin": neutral_grid,
            "n_candidates": len(candidates),
        },
        "current_params": current_params,
        "current_valid": current_valid,
        "current_test": current_test,
        "selected_params": selected["params"],
        "selected_valid_score": selected["valid_score"],
        "selected_valid_metrics": selected["valid_metrics"],
        "selected_test": test_result,
        "passed": passed,
        "warnings": warnings,
        "top_valid_candidates": top,
    }


def trust_gate(snap: PipelineSnapshot, segments: Dict[str, dict]) -> Dict[str, Any]:
    meta = snap.model.meta
    test_seg = segments.get("test", {})
    test_metrics = test_seg.get("metrics", {}) if test_seg.get("available") else {}
    test_summary = test_metrics.get("trade_summary", {})

    reasons: list[str] = []
    warnings: list[str] = []

    pf = float(test_metrics.get("profit_factor", 0.0) or 0.0)
    total_return = float(test_metrics.get("total_return", 0.0) or 0.0)
    sharpe = float(test_metrics.get("sharpe", 0.0) or 0.0)
    max_dd = float(test_metrics.get("max_drawdown", 0.0) or 0.0)
    n_trades = int(test_metrics.get("n_trades", 0) or 0)
    lift = float(getattr(meta, "accuracy_lift_vs_baseline", 0.0) or 0.0) if meta else 0.0

    if not test_seg.get("available"):
        reasons.append("test-only backtest is unavailable")
    if n_trades < 5:
        reasons.append(f"too few test trades: {n_trades}")
    if total_return <= 0:
        reasons.append(f"test-only return is not positive: {total_return:.2f}%")
    if pf < 1.10:
        reasons.append(f"test-only profit factor is below 1.10: {pf:.2f}")
    if sharpe <= 0:
        reasons.append(f"test-only Sharpe is not positive: {sharpe:.2f}")
    if max_dd < -25:
        reasons.append(f"test-only max drawdown is too deep: {max_dd:.2f}%")
    if lift < 0:
        warnings.append(f"model accuracy is below baseline by {abs(lift) * 100:.2f} percentage points")
    elif lift < 0.01:
        warnings.append(f"model accuracy lift is weak: {lift * 100:.2f} percentage points")

    enabled = len(reasons) == 0
    level = "trusted" if enabled and not warnings else ("watch" if enabled else "disabled")

    return {
        "enabled": enabled,
        "level": level,
        "message": "timeframe passed test-only edge gate" if enabled else "timeframe disabled: no confirmed test-only edge",
        "rules": {
            "min_test_trades": 5,
            "min_test_return_pct": 0.0,
            "min_test_profit_factor": 1.10,
            "min_test_sharpe": 0.0,
            "max_allowed_test_drawdown_pct": -25.0,
        },
        "test_only": {
            "total_return": total_return,
            "profit_factor": pf,
            "sharpe": sharpe,
            "max_drawdown": max_dd,
            "n_trades": n_trades,
            "net_pnl": test_summary.get("net_pnl"),
            "equity_pnl": test_summary.get("equity_pnl"),
        },
        "model_lift_vs_baseline": lift,
        "reasons": reasons,
        "warnings": warnings,
    }
