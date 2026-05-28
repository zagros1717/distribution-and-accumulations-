"""
btcalpha.api.walkforward_sides
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Embargoed walk-forward side audit.

This diagnostic answers one narrow question:
  Is the current timeframe/model failing on both sides, or mainly because of
  long/short exposure? It trains one temporary model per fold, generates the
  normal strategy decisions, then replays the same fold three ways:
    - long_short
    - long_only
    - short_only

It is diagnostic only. It does not modify the production model or config.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict

import numpy as np
import pandas as pd

from btcalpha.backtest.engine import Backtester
from btcalpha.live.engine import PipelineSnapshot
from btcalpha.strategy.signal import Strategy
from btcalpha.api.evaluation import (
    _baseline_accuracy,
    _bt_to_payload,
    _filter_decisions,
    _fit_temp_model,
    _wf_windows,
)

_MODES = ("long_short", "long_only", "short_only")


def _empty(reason: str) -> dict:
    return {"available": False, "reason": reason, "summary": {}, "modes": {}}


def _summarize_mode(folds: list[dict]) -> dict:
    ok = [f for f in folds if f.get("available")]
    if not ok:
        return {"available": False, "reason": "no successful folds"}

    returns = [float(f["metrics"].get("total_return", 0.0) or 0.0) for f in ok]
    pfs = [float(f["metrics"].get("profit_factor", 0.0) or 0.0) for f in ok]
    sharpes = [float(f["metrics"].get("sharpe", 0.0) or 0.0) for f in ok]
    max_dds = [float(f["metrics"].get("max_drawdown", 0.0) or 0.0) for f in ok]
    n_trades = [int(f["metrics"].get("n_trades", 0) or 0) for f in ok]
    long_pnl = [float(f["metrics"].get("trade_summary", {}).get("long_pnl", 0.0) or 0.0) for f in ok]
    short_pnl = [float(f["metrics"].get("trade_summary", {}).get("short_pnl", 0.0) or 0.0) for f in ok]

    positive_return = [r for r in returns if r > 0]
    positive_pf = [pf for pf in pfs if pf >= 1.10]
    acceptable_dd = [dd for dd in max_dds if dd >= -25.0]
    min_required = max(1, int(np.ceil(len(ok) * 0.6)))
    passed = (
        len(positive_return) >= min_required
        and len(positive_pf) >= min_required
        and float(np.nanmean(returns)) > 0
        and float(np.nanmean(pfs)) >= 1.10
    )

    warnings = []
    if len(positive_return) < min_required:
        warnings.append("positive return in too few folds")
    if len(positive_pf) < min_required:
        warnings.append("profit factor >= 1.10 in too few folds")
    if float(np.nanmean(returns)) <= 0:
        warnings.append("average return is not positive")
    if float(np.nanmean(pfs)) < 1.10:
        warnings.append("average profit factor is below 1.10")
    if len(acceptable_dd) < min_required:
        warnings.append("drawdown is deeper than -25% in too many folds")

    return {
        "available": True,
        "passed": passed,
        "n_available_folds": len(ok),
        "positive_return_folds": len(positive_return),
        "positive_profit_factor_folds": len(positive_pf),
        "acceptable_drawdown_folds": len(acceptable_dd),
        "avg_return": round(float(np.nanmean(returns)), 3),
        "avg_profit_factor": round(float(np.nanmean(pfs)), 3),
        "avg_sharpe": round(float(np.nanmean(sharpes)), 3),
        "avg_max_drawdown": round(float(np.nanmean(max_dds)), 3),
        "avg_n_trades": round(float(np.nanmean(n_trades)), 1),
        "sum_long_pnl": round(float(np.nansum(long_pnl)), 2),
        "sum_short_pnl": round(float(np.nansum(short_pnl)), 2),
        "warnings": warnings,
    }


def _rank_modes(mode_summaries: dict) -> list[dict]:
    rows = []
    for mode, payload in mode_summaries.items():
        summary = payload.get("summary", {}) if isinstance(payload, dict) else {}
        if not summary.get("available"):
            continue
        rows.append({
            "mode": mode,
            "passed": bool(summary.get("passed", False)),
            "avg_return": summary.get("avg_return"),
            "avg_profit_factor": summary.get("avg_profit_factor"),
            "positive_return_folds": summary.get("positive_return_folds"),
            "positive_profit_factor_folds": summary.get("positive_profit_factor_folds"),
            "avg_max_drawdown": summary.get("avg_max_drawdown"),
        })
    rows.sort(key=lambda r: (
        bool(r.get("passed")),
        float(r.get("avg_profit_factor") or -999),
        float(r.get("avg_return") or -999),
    ), reverse=True)
    return rows


def _recommendation(ranked: list[dict]) -> dict:
    if not ranked:
        return {"action": "no_decision", "text": "No successful side walk-forward modes were available."}
    best = ranked[0]
    if best.get("passed"):
        return {
            "action": "research_candidate",
            "mode": best["mode"],
            "text": f"{best['mode']} passed the side walk-forward gate; treat it as a research candidate, not automatic live approval.",
        }
    return {
        "action": "do_not_trade_timeframe",
        "best_mode": best["mode"],
        "text": "No side mode passed embargoed walk-forward. Keep executable live trading disabled for this timeframe.",
    }


def walk_forward_side_audit(snap: PipelineSnapshot, n_folds: int = 4) -> Dict[str, Any]:
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
        return _empty("not enough labeled rows for side walk-forward audit")

    mode_folds: dict[str, list[dict]] = {m: [] for m in _MODES}
    model_folds: list[dict] = []

    for w in windows:
        tr0, tr1 = w["train_pos"]
        te0, te1 = w["test_pos"]
        train_idx = labeled_idx[tr0:tr1]
        test_idx = labeled_idx[te0:te1]
        Xtr, ytr = X.loc[train_idx], y.loc[train_idx]
        Xte, yte = X.loc[test_idx], y.loc[test_idx]
        fold_common = {k: v for k, v in w.items() if not k.endswith("_pos")}
        try:
            temp_model = _fit_temp_model(snap.model.timeframe, Xtr, ytr)
            proba = temp_model.predict_proba(Xte)
            pred = proba[["p_down", "p_neutral", "p_up"]].to_numpy().argmax(axis=1)
            acc = float((pred == yte.to_numpy()).mean()) if len(yte) else 0.0
            majority, base_acc = _baseline_accuracy(ytr, yte)
            lift = acc - base_acc

            raw = snap.features["raw"].reindex(test_idx).dropna(subset=["open", "high", "low", "close"])
            proba = proba.reindex(raw.index)
            regime = snap.regime.reindex(raw.index).ffill()
            regime["regime_label"] = regime["regime_label"].fillna("neutral")
            regime["regime_score"] = regime["regime_score"].fillna(0.0)
            decisions = Strategy(snap.model.timeframe).decide_series(proba, regime, raw)

            model_folds.append({
                **fold_common,
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
            })

            for mode in _MODES:
                mode_decisions = _filter_decisions(decisions, mode)
                bt = Backtester().run(mode_decisions, raw)
                mode_folds[mode].append({
                    **fold_common,
                    "available": True,
                    "n_train": int(len(Xtr)),
                    "n_test": int(len(Xte)),
                    "metrics": _bt_to_payload(bt),
                })
        except Exception as exc:  # noqa: BLE001
            err = {**fold_common, "available": False, "error": str(exc), "n_train": int(len(Xtr)), "n_test": int(len(Xte))}
            model_folds.append(err)
            for mode in _MODES:
                mode_folds[mode].append(dict(err))

    modes_payload = {}
    for mode in _MODES:
        folds = mode_folds[mode]
        modes_payload[mode] = {
            "summary": _summarize_mode(folds),
            "folds": folds,
        }

    ranked = _rank_modes(modes_payload)
    ok_model_folds = [f for f in model_folds if f.get("available")]
    lifts = [float(f.get("lift_vs_baseline", 0.0) or 0.0) for f in ok_model_folds]

    return {
        "available": bool(ok_model_folds),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "method": "expanding_walk_forward_with_embargo_side_modes",
        "n_requested_folds": int(n_folds),
        "n_available_folds": int(len(ok_model_folds)),
        "embargo_bars": int(max(1, horizon)),
        "class_weighting": ok_model_folds[0].get("class_weighting") if ok_model_folds else None,
        "model_lift_summary": {
            "positive_model_lift_folds": len([x for x in lifts if x > 0]),
            "avg_lift_vs_baseline": round(float(np.nanmean(lifts)), 6) if lifts else None,
        },
        "ranked_modes": ranked,
        "recommendation": _recommendation(ranked),
        "modes": modes_payload,
        "model_folds": model_folds,
    }
