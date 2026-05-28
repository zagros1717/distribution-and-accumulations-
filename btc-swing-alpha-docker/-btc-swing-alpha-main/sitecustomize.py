"""
Runtime safety patches for BTC Swing Alpha research API.

Put this file at the repository root, commit, push, and let Railway redeploy.

What it fixes at runtime:
1) Trust Gate is advisory only; it does not overwrite raw alpha direction/position.
2) API backtest split reconstruction becomes purge-aware instead of contiguous.
3) Feature engineering exposes labeled_index and row-count audit fields.
4) needs_retrain handles timezone-aware and naive trained_at values safely.

This is a safe overlay patch. Later, these changes should be folded directly into
btcalpha/api/server.py, btcalpha/api/evaluation.py, btcalpha/features/engineering.py,
and btcalpha/model/alpha_model.py.
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone


def _patch_features() -> None:
    try:
        import numpy as np
        import pandas as pd
        from btcalpha.config import get_config, get_logger
        from btcalpha.features import engineering as eng
    except Exception:
        return

    log = get_logger("features")

    def build_features(dataset: pd.DataFrame, timeframe: str) -> dict:
        cfg = get_config()
        log.info("ساخت فیچر برای %s ...", timeframe)

        tech = eng._technical_features(dataset, cfg)
        micro = eng._microstructure_features(dataset, timeframe)
        macro = eng._macro_features(dataset)
        include_macro = eng._include_macro_features(cfg)

        parts = [tech, micro]
        if include_macro:
            parts.append(macro)
            log.warning(
                "MACRO FEATURES ENABLED — this changes the ML signal; macro_features=%d",
                len(macro.columns),
            )
        else:
            log.info(
                "ماکرو فقط context/regime است و از فیچرهای مدل حذف شد | macro_candidates=%d",
                len(macro.columns),
            )

        X = pd.concat(parts, axis=1).replace([np.inf, -np.inf], np.nan)
        y, label_audit = eng.make_labels(dataset, timeframe, cfg)

        feature_valid = X.dropna().index
        X = X.loc[feature_valid]
        y = y.loc[feature_valid]
        labeled_index = y.dropna().index

        feature_cols = list(X.columns)
        audit = eng._feature_audit(feature_cols, len(macro.columns), include_macro, label_audit)
        audit["n_feature_valid_rows"] = int(len(X))
        audit["n_labeled_rows"] = int(len(labeled_index))
        audit["n_trailing_unlabeled_rows"] = int(len(X) - len(labeled_index))

        log.info(
            "تعداد فیچر: %d | ردیف feature-valid: %d | ردیف labeled: %d | macro_used=%d | label=%s",
            len(feature_cols),
            len(X),
            len(labeled_index),
            audit["macro_features_used"],
            label_audit.get("label_mode"),
        )

        return {
            "X": X,
            "y": y,
            "labeled_index": labeled_index,
            "feature_cols": feature_cols,
            "feature_audit": audit,
            "label_audit": label_audit,
            "raw": dataset.loc[feature_valid],
            "timeframe": timeframe,
        }

    eng.build_features = build_features


def _patch_model() -> None:
    try:
        from btcalpha.model.alpha_model import AlphaModel
    except Exception:
        return

    def needs_retrain(self) -> bool:
        if not self.cfg["model"]["retrain"]["enabled"]:
            return False
        if self.meta is None:
            return True

        trained = datetime.fromisoformat(self.meta.trained_at)
        if trained.tzinfo is None:
            trained = trained.replace(tzinfo=timezone.utc)

        age_days = (datetime.now(timezone.utc) - trained).days
        return age_days >= self.cfg["model"]["retrain"]["every_n_days"]

    AlphaModel.needs_retrain = needs_retrain


def _patch_evaluation() -> None:
    try:
        from btcalpha.api import evaluation
    except Exception:
        return

    def _split_indices(snap) -> dict:
        meta = snap.model.meta
        if meta is None:
            return {}

        labeled_idx = snap.features.get("labeled_index")
        if labeled_idx is None:
            y = snap.features["y"]
            labeled_idx = y[y.notna()].index

        n = len(labeled_idx)
        cfg = snap.model.cfg
        h = max(
            0,
            int(cfg.get("features", {}).get("horizons", {}).get(snap.model.timeframe, 0) or 0),
        )

        tr = int(n * cfg["model"]["train_ratio"])
        va = int(n * (cfg["model"]["train_ratio"] + cfg["model"].get("valid_ratio", 0.0)))

        train_end = max(0, tr - h)
        valid_start = min(n, tr)
        valid_end = max(valid_start, va - h)
        test_start = min(n, va)

        train_end = max(0, min(int(train_end), n))
        valid_start = max(0, min(int(valid_start), n))
        valid_end = max(valid_start, min(int(valid_end), n))
        test_start = max(valid_end, min(int(test_start), n))

        return {
            "train": labeled_idx[:train_end],
            "valid": labeled_idx[valid_start:valid_end],
            "test": labeled_idx[test_start:],
            "split": {
                "n_train": int(train_end),
                "n_valid": int(max(0, valid_end - valid_start)),
                "n_test": int(max(0, n - test_start)),
                "labeled_rows": int(n),
                "purged_between_train_valid": int(max(0, valid_start - train_end)),
                "purged_between_valid_test": int(max(0, test_start - valid_end)),
                "split_method": "chronological_purged_horizon_embargo",
                "split_positions_source": "runtime_recomputed",
            },
        }

    evaluation._split_indices = _split_indices


def _patch_server() -> None:
    try:
        from btcalpha.api import server
    except Exception:
        return

    def _apply_trust_gate_to_live(live: dict, gate: dict) -> dict:
        """
        Research-first behavior:
        Trust Gate is advisory only. It must never rewrite raw alpha direction,
        position, probabilities, bias, stop loss, or take profit.
        """
        out = copy.deepcopy(live) if isinstance(live, dict) else {}

        gate_passed = bool(gate.get("enabled", False))
        reasons = gate.get("reasons") or []
        warnings = gate.get("warnings") or []

        out["trust_gate_advisory"] = gate
        out["trust_gate_blocked_live"] = False
        out["alpha_status"] = {
            "status": "alpha_candidate" if gate_passed else "no_confirmed_alpha",
            "research_only": True,
            "raw_signal_is_not_blocked": True,
            "gate_passed": gate_passed,
            "notes": reasons + warnings,
        }

        return out

    server._apply_trust_gate_to_live = _apply_trust_gate_to_live

    try:
        from btcalpha.api.down_edge_benchmark import down_edge_benchmark_audit

        @server.app.get("/api/audit/{tf}/down-edge-benchmark")
        def get_down_edge_benchmark(tf: str, force: bool = False, folds: int = 4):
            tf = server._valid_tf(tf)
            snap = server._get_snapshot(tf, force=force)
            return server._json({
                "timeframe": tf,
                "generated_at": snap.generated_at,
                "model_audit": server._model_audit(snap),
                "down_edge_benchmark": down_edge_benchmark_audit(snap, n_folds=folds),
            })
    except Exception:
        pass

    try:
        server.app.version = "1.20-alpha-research-patched"
    except Exception:
        pass


def _apply() -> None:
    # Order matters:
    # features/model/evaluation must be patched before server imports live.engine
    # and before requests call run_pipeline().
    _patch_features()
    _patch_model()
    _patch_evaluation()
    _patch_server()


_apply()