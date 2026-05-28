"""
btcalpha.api.long_edge_benchmark
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Diagnostic benchmark for binary long-edge models.

Research-only endpoint. It does not modify production models, config, live
signals, or Trust Gate. It compares temporary binary classifiers using an
expanding walk-forward split with an embargo.

Important: the pass gate is intentionally strict. A model cannot pass because of
one or two tiny-trade-count lucky folds. It must be stable across folds, have
minimum trade counts, avoid large bad folds, and beat the naive binary baseline
on classification accuracy in most folds.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict

import numpy as np
import pandas as pd

from btcalpha.api.evaluation import _bt_to_payload, _wf_windows
from btcalpha.backtest.engine import Backtester
from btcalpha.config import get_config
from btcalpha.features.engineering import (
    _atr,
    _first_trade_exit,
    _trade_outcome_min_edge,
    _trade_outcome_pnl_pct,
)
from btcalpha.live.engine import PipelineSnapshot


STRICT_GATE = {
    "min_trades_per_fold": 20,
    "min_avg_return_pct": 5.0,
    "min_avg_profit_factor": 1.20,
    "max_allowed_bad_fold_return_pct": -10.0,
    "min_positive_return_fold_ratio": 0.75,
    "min_positive_profit_factor_fold_ratio": 0.75,
    "min_accuracy_above_baseline_fold_ratio": 0.75,
    "max_avg_drawdown_pct": -25.0,
}


@dataclass
class _BinaryFit:
    name: str
    model: Any
    feature_cols: list[str]

    def predict_proba_one(self, X: pd.DataFrame) -> pd.Series:
        Xp = X[self.feature_cols]
        proba = self.model.predict_proba(Xp)
        classes = list(getattr(self.model, "classes_", [0, 1]))
        if 1 in classes:
            arr = proba[:, classes.index(1)]
        else:
            arr = np.zeros(len(Xp), dtype=float)
        return pd.Series(arr, index=X.index, name="p_long_edge")


def _binary_long_labels(raw: pd.DataFrame, timeframe: str) -> tuple[pd.Series, pd.Series, dict]:
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
        long_entry = entry_open * (1.0 + slip)
        long_stop = cur_close - stop_atr * cur_atr
        long_take = cur_close + take_atr * cur_atr
        long_exit, _ = _first_trade_exit(raw, entry_i, end_i, long_stop, long_take, "long", slip)
        lp = _trade_outcome_pnl_pct(long_entry, long_exit, fee, "long")
        pnl_pct.iloc[i] = lp
        y.iloc[i] = 1.0 if lp > min_edge else 0.0

    valid = pnl_pct.dropna()
    audit = {
        "label": "binary_long_edge",
        "positive_definition": "long trade pnl_pct > min_edge after fees/slippage/SL/TP",
        "horizon": horizon,
        "min_edge_pct": round(min_edge * 100, 4),
        "fee_pct": round(fee * 100, 4),
        "slippage_pct": round(slip * 100, 4),
        "stop_loss_atr": stop_atr,
        "take_profit_atr": take_atr,
        "valid_labels": int(y.notna().sum()),
        "positive_rate": round(float(y.dropna().mean()), 4) if y.notna().any() else None,
        "avg_long_outcome_pct": round(float(valid.mean() * 100), 4) if len(valid) else None,
    }
    return y, pnl_pct, audit


def _class_dist(y: pd.Series) -> dict:
    if y is None or len(y) == 0:
        return {}
    vc = y.astype(int).value_counts().sort_index()
    return {int(k): int(v) for k, v in vc.items()}


def _make_estimator(kind: str):
    kind = kind.lower().strip()
    if kind == "lightgbm":
        try:
            import lightgbm as lgb
        except Exception as exc:  # noqa: BLE001
            return None, f"lightgbm unavailable: {exc}"
        return lgb.LGBMClassifier(
            objective="binary",
            n_estimators=250,
            learning_rate=0.025,
            max_depth=5,
            num_leaves=31,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_samples=50,
            reg_alpha=0.2,
            reg_lambda=0.2,
            random_state=42,
            n_jobs=1,
            verbosity=-1,
        ), None

    if kind == "xgboost":
        try:
            from xgboost import XGBClassifier
        except Exception as exc:  # noqa: BLE001
            return None, f"xgboost unavailable: {exc}"
        return XGBClassifier(
            objective="binary:logistic",
            n_estimators=250,
            learning_rate=0.025,
            max_depth=4,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=10,
            reg_alpha=0.2,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=1,
            eval_metric="logloss",
            tree_method="hist",
        ), None

    return None, f"unknown estimator: {kind}"


def _fit_binary(kind: str, X: pd.DataFrame, y: pd.Series, feature_cols: list[str]) -> tuple[_BinaryFit | None, str | None]:
    if len(y.astype(int).unique()) < 2:
        return None, f"not enough class variety: {_class_dist(y)}"
    estimator, err = _make_estimator(kind)
    if estimator is None:
        return None, err
    estimator.fit(X[feature_cols], y.astype(int))
    return _BinaryFit(name=kind, model=estimator, feature_cols=feature_cols), None


def _decisions_from_prob(raw: pd.DataFrame, prob: pd.Series, threshold: float, timeframe: str) -> pd.DataFrame:
    cfg = get_config()
    stop_atr = float(cfg.get("strategy", {}).get("stop_loss_atr", 2.0))
    take_atr = float(cfg.get("strategy", {}).get("take_profit_atr", 4.5))
    atr = _atr(raw, int(cfg["features"].get("atr_window", 14))).reindex(raw.index)
    prob = prob.reindex(raw.index).fillna(0.0)

    out = pd.DataFrame(index=raw.index)
    out["direction"] = np.where(prob >= threshold, "long", "flat")
    out["raw_alpha"] = prob - threshold
    out["final_signal"] = prob - threshold
    out["position"] = np.where(prob >= threshold, 1.0, 0.0)
    out["confidence"] = prob
    out["regime"] = "neutral"
    out["regime_score"] = 0.0
    out["trend_strength"] = np.nan
    out["stop_loss"] = np.where(out["direction"] == "long", raw["close"] - stop_atr * atr, np.nan)
    out["take_profit"] = np.where(out["direction"] == "long", raw["close"] + take_atr * atr, np.nan)
    out["reasons"] = [[f"binary_long_edge p={float(p):.3f} threshold={threshold:.2f}"] for p in prob]
    out.attrs["timeframe"] = timeframe
    return out


def _score_metrics(metrics: dict, min_trades: int = 10) -> float:
    n = int(metrics.get("n_trades", 0) or 0)
    if n < min_trades:
        return -1e9
    total_return = float(metrics.get("total_return", 0.0) or 0.0)
    pf = float(metrics.get("profit_factor", 0.0) or 0.0)
    sharpe = float(metrics.get("sharpe", 0.0) or 0.0)
    max_dd = float(metrics.get("max_drawdown", 0.0) or 0.0)
    return total_return + 12.0 * (pf - 1.0) + 5.0 * sharpe + 0.25 * max_dd


def _eval_threshold(raw: pd.DataFrame, prob: pd.Series, threshold: float, timeframe: str) -> dict:
    decisions = _decisions_from_prob(raw, prob, threshold, timeframe)
    bt = Backtester().run(decisions, raw)
    return _bt_to_payload(bt)


def _inner_split(idx: pd.Index, horizon: int) -> tuple[pd.Index, pd.Index]:
    n = len(idx)
    cut = int(n * 0.80)
    train_end = max(0, cut - horizon)
    valid_start = min(n, cut)
    return idx[:train_end], idx[valid_start:]


def _select_threshold(kind: str, X: pd.DataFrame, y: pd.Series, raw: pd.DataFrame, train_idx: pd.Index, timeframe: str, thresholds: list[float], horizon: int, feature_cols: list[str]) -> dict:
    subtrain_idx, inner_valid_idx = _inner_split(train_idx, horizon)
    if len(subtrain_idx) < 300 or len(inner_valid_idx) < 40 or y.loc[subtrain_idx].nunique() < 2:
        return {
            "selected_threshold": 0.55,
            "reason": "inner validation unavailable; using conservative default",
            "candidates": [],
        }

    fit, err = _fit_binary(kind, X.loc[subtrain_idx], y.loc[subtrain_idx], feature_cols)
    if fit is None:
        return {"selected_threshold": 0.55, "reason": err, "candidates": []}

    raw_inner = raw.reindex(inner_valid_idx).dropna(subset=["open", "high", "low", "close"])
    prob_inner = fit.predict_proba_one(X.loc[raw_inner.index])
    candidates = []
    for thr in thresholds:
        metrics = _eval_threshold(raw_inner, prob_inner, thr, timeframe)
        score = _score_metrics(metrics, min_trades=5)
        candidates.append({
            "threshold": float(thr),
            "score": round(float(score), 4),
            "metrics": metrics,
        })
    candidates.sort(key=lambda c: c["score"], reverse=True)
    return {
        "selected_threshold": float(candidates[0]["threshold"]),
        "reason": "selected on inner validation only",
        "candidates": candidates,
    }


def _binary_accuracy(prob: pd.Series, y: pd.Series, threshold: float = 0.5) -> float:
    pred = (prob >= threshold).astype(int)
    return float((pred == y.astype(int)).mean()) if len(y) else 0.0


def _required_folds(n: int, ratio_key: str) -> int:
    return max(1, int(np.ceil(n * float(STRICT_GATE[ratio_key]))))


def _strict_pass_details(ok_folds: list[dict]) -> tuple[bool, list[str], dict]:
    n = len(ok_folds)
    min_required_return = _required_folds(n, "min_positive_return_fold_ratio")
    min_required_pf = _required_folds(n, "min_positive_profit_factor_fold_ratio")
    min_required_acc = _required_folds(n, "min_accuracy_above_baseline_fold_ratio")

    returns = [float(f["metrics"].get("total_return", 0.0) or 0.0) for f in ok_folds]
    pfs = [float(f["metrics"].get("profit_factor", 0.0) or 0.0) for f in ok_folds]
    max_dds = [float(f["metrics"].get("max_drawdown", 0.0) or 0.0) for f in ok_folds]
    n_trades = [int(f["metrics"].get("n_trades", 0) or 0) for f in ok_folds]

    positive_return_folds = len([r for r in returns if r > 0])
    positive_pf_folds = len([pf for pf in pfs if pf >= 1.10])
    min_trade_folds = len([x for x in n_trades if x >= int(STRICT_GATE["min_trades_per_fold"])])
    no_large_bad_fold = all(r >= float(STRICT_GATE["max_allowed_bad_fold_return_pct"]) for r in returns)
    avg_return_ok = float(np.nanmean(returns)) >= float(STRICT_GATE["min_avg_return_pct"])
    avg_pf_ok = float(np.nanmean(pfs)) >= float(STRICT_GATE["min_avg_profit_factor"])
    avg_dd_ok = float(np.nanmean(max_dds)) >= float(STRICT_GATE["max_avg_drawdown_pct"])
    acc_above_baseline_folds = len([
        f for f in ok_folds
        if float(f.get("binary_accuracy_at_0_5", 0.0) or 0.0) > float(f.get("baseline_accuracy", 0.0) or 0.0)
    ])

    checks = {
        "min_trade_folds": min_trade_folds,
        "required_min_trade_folds": n,
        "positive_return_folds": positive_return_folds,
        "required_positive_return_folds": min_required_return,
        "positive_profit_factor_folds": positive_pf_folds,
        "required_positive_profit_factor_folds": min_required_pf,
        "accuracy_above_baseline_folds": acc_above_baseline_folds,
        "required_accuracy_above_baseline_folds": min_required_acc,
        "no_fold_return_below_limit": no_large_bad_fold,
        "avg_return_ok": avg_return_ok,
        "avg_profit_factor_ok": avg_pf_ok,
        "avg_drawdown_ok": avg_dd_ok,
    }

    warnings = []
    if min_trade_folds < n:
        warnings.append(f"some folds have fewer than {STRICT_GATE['min_trades_per_fold']} trades")
    if positive_return_folds < min_required_return:
        warnings.append("positive return in too few folds")
    if positive_pf_folds < min_required_pf:
        warnings.append("profit factor >= 1.10 in too few folds")
    if acc_above_baseline_folds < min_required_acc:
        warnings.append("binary accuracy does not beat baseline in enough folds")
    if not no_large_bad_fold:
        warnings.append(f"at least one fold return is below {STRICT_GATE['max_allowed_bad_fold_return_pct']}%")
    if not avg_return_ok:
        warnings.append(f"average return is below {STRICT_GATE['min_avg_return_pct']}%")
    if not avg_pf_ok:
        warnings.append(f"average profit factor is below {STRICT_GATE['min_avg_profit_factor']}")
    if not avg_dd_ok:
        warnings.append(f"average drawdown is deeper than {STRICT_GATE['max_avg_drawdown_pct']}%")

    passed = (
        min_trade_folds == n
        and positive_return_folds >= min_required_return
        and positive_pf_folds >= min_required_pf
        and acc_above_baseline_folds >= min_required_acc
        and no_large_bad_fold
        and avg_return_ok
        and avg_pf_ok
        and avg_dd_ok
    )
    return passed, warnings, checks


def _summarize_folds(folds: list[dict]) -> dict:
    ok = [f for f in folds if f.get("available")]
    if not ok:
        return {"available": False, "reason": "no successful folds"}
    returns = [float(f["metrics"].get("total_return", 0.0) or 0.0) for f in ok]
    pfs = [float(f["metrics"].get("profit_factor", 0.0) or 0.0) for f in ok]
    sharpes = [float(f["metrics"].get("sharpe", 0.0) or 0.0) for f in ok]
    max_dds = [float(f["metrics"].get("max_drawdown", 0.0) or 0.0) for f in ok]

    strict_passed, strict_warnings, strict_checks = _strict_pass_details(ok)
    legacy_min_required = max(1, int(np.ceil(len(ok) * 0.60)))
    legacy_positive = len([r for r in returns if r > 0])
    legacy_good_pf = len([pf for pf in pfs if pf >= 1.10])
    legacy_passed = (
        legacy_positive >= legacy_min_required
        and legacy_good_pf >= legacy_min_required
        and float(np.nanmean(returns)) > 0
        and float(np.nanmean(pfs)) >= 1.10
    )

    return {
        "available": True,
        "passed": strict_passed,
        "legacy_loose_passed": legacy_passed,
        "n_available_folds": len(ok),
        "positive_return_folds": legacy_positive,
        "positive_profit_factor_folds": legacy_good_pf,
        "accuracy_above_baseline_folds": strict_checks["accuracy_above_baseline_folds"],
        "min_trade_folds": strict_checks["min_trade_folds"],
        "avg_return": round(float(np.nanmean(returns)), 3),
        "avg_profit_factor": round(float(np.nanmean(pfs)), 3),
        "avg_sharpe": round(float(np.nanmean(sharpes)), 3),
        "avg_max_drawdown": round(float(np.nanmean(max_dds)), 3),
        "strict_gate": STRICT_GATE,
        "strict_checks": strict_checks,
        "warnings": strict_warnings,
    }


def _rank_models(results: dict) -> list[dict]:
    rows = []
    for name, payload in results.items():
        s = payload.get("summary", {})
        if not s.get("available"):
            continue
        rows.append({
            "model": name,
            "passed": bool(s.get("passed", False)),
            "legacy_loose_passed": bool(s.get("legacy_loose_passed", False)),
            "avg_return": s.get("avg_return"),
            "avg_profit_factor": s.get("avg_profit_factor"),
            "positive_return_folds": s.get("positive_return_folds"),
            "positive_profit_factor_folds": s.get("positive_profit_factor_folds"),
            "accuracy_above_baseline_folds": s.get("accuracy_above_baseline_folds"),
            "avg_max_drawdown": s.get("avg_max_drawdown"),
        })
    rows.sort(key=lambda r: (r["passed"], float(r["avg_profit_factor"] or -999), float(r["avg_return"] or -999)), reverse=True)
    return rows


def long_edge_benchmark_audit(snap: PipelineSnapshot, n_folds: int = 4) -> Dict[str, Any]:
    timeframe = snap.timeframe
    X_all = snap.features["X"].copy()
    raw_all = snap.features["raw"].copy()
    y_all, _pnl_pct, label_audit = _binary_long_labels(raw_all, timeframe)

    common_idx = X_all.index.intersection(y_all.dropna().index)
    X = X_all.loc[common_idx].copy()
    y = y_all.loc[common_idx].astype(int).copy()
    raw = raw_all.reindex(common_idx)
    labeled_idx = X.index
    horizon = int(label_audit["horizon"])
    windows = _wf_windows(labeled_idx, horizon=max(1, horizon), n_folds=n_folds)
    thresholds = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    feature_cols = [c for c in list(snap.features.get("feature_cols", X.columns)) if c in X.columns]
    estimators = ["lightgbm", "xgboost"]

    if not windows:
        return {"available": False, "reason": "not enough labeled rows", "label_audit": label_audit}

    results: dict[str, dict] = {}
    for kind in estimators:
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
                folds.append({
                    **fold_common,
                    "available": True,
                    "n_train": int(len(train_idx)),
                    "n_test": int(len(test_idx)),
                    "train_class_distribution": _class_dist(y.loc[train_idx]),
                    "test_class_distribution": _class_dist(y.loc[test_idx]),
                    "selected_threshold": selected_threshold,
                    "threshold_selection": threshold_info,
                    "binary_accuracy_at_0_5": round(acc, 6),
                    "baseline_accuracy": round(baseline, 6),
                    "accuracy_lift_vs_baseline": round(acc - baseline, 6),
                    "metrics": metrics,
                })
            except Exception as exc:  # noqa: BLE001
                folds.append({**fold_common, "available": False, "error": str(exc), "n_train": int(len(train_idx)), "n_test": int(len(test_idx))})
        results[kind] = {"summary": _summarize_folds(folds), "folds": folds}

    ranked = _rank_models(results)
    recommendation = {
        "action": "do_not_enable_live",
        "text": "This is a strict research benchmark only. 1d executable trading remains disabled unless a model passes strict walk-forward, calibration, and Trust Gate criteria.",
        "best_model": ranked[0]["model"] if ranked else None,
    }
    if ranked and ranked[0].get("passed"):
        recommendation["action"] = "research_candidate_only"
        recommendation["text"] = "A binary long-edge model passed the strict diagnostic. Keep 1d live disabled until it is separately integrated, calibrated, and gated."

    return {
        "available": True,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "method": "strict_binary_long_edge_inner_threshold_expanding_walk_forward_with_embargo",
        "timeframe": timeframe,
        "embargo_bars": int(max(1, horizon)),
        "threshold_grid": thresholds,
        "strict_gate": STRICT_GATE,
        "label_audit": label_audit,
        "ranked_models": ranked,
        "recommendation": recommendation,
        "models": results,
    }
