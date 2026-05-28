"""
XGBoost trainer with walk-forward validation.

Inputs:
  - feature parquet partition for a horizon's interval_ms (1000ms by default)
  - label parquet partition for the chosen horizon

Pipeline:
  1. Load features and labels, join on (exchange, symbol, ts).
  2. Drop rows where the snapshot was corrupt (is_valid=False).
  3. Iterate walk-forward windows:
        train: [t0, t0 + train_days)
        val:   [t0 + train_days, t0 + train_days + val_days)
        step:  +step_days
     For each window:
        - assert temporal order (no leakage)
        - train XGBoost with early stopping on val
        - record metrics: accuracy / per-class precision / log-loss / confusion matrix
        - capture feature importance (gain)
        - capture OOS predictions (val rows) for the downstream backtester
  4. Return a list of fold results + the last-fold model. Save per-fold OOS
     predictions to `oos_predictions.parquet` next to the model so the backtest
     can run in true OOS mode without re-doing inference.

NOTE: no feature scaling is applied. XGBoost is invariant to monotonic
transformations of inputs, so a StandardScaler-style step would have no
model benefit while creating a save/load consistency hazard (a scaler fit
on train must be carried into inference; forgetting that gives silent
garbage predictions). See src/models/evaluate.py for the inference path.

Anti-leakage controls live in src/utils/validation.py.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, log_loss,
)

from src.storage.parquet_store import features_path, labels_path, read_parquet_dir
from src.utils.logging import logger
from src.utils.validation import assert_train_before_val, assert_monotonic_time


# Columns that are NOT features (they identify the row or are derived from
# information at-or-after T's mid price).
NON_FEATURE_COLS = {
    "ts", "exchange", "symbol", "horizon_s",
    "future_mid_price", "future_return", "label", "label_class",
    "is_valid", "threshold_bps",
}


@dataclass
class FoldResult:
    fold_index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    n_train: int
    n_val: int
    accuracy: float
    log_loss: float
    precision_long: float
    precision_short: float
    recall_long: float
    recall_short: float
    f1_long: float
    f1_short: float
    n_signals_long: int
    n_signals_short: int
    confusion: List[List[int]]
    feature_importance_gain: Dict[str, float]


@dataclass
class TrainingResult:
    horizon_s: int
    interval_ms: int
    exchange: str
    symbol: str
    feature_columns: List[str]
    folds: List[FoldResult] = field(default_factory=list)
    final_model_path: Optional[str] = None
    # Parquet file containing one row per OOS prediction (from validation
    # windows only). The backtester reads this directly so it is guaranteed
    # out-of-sample. See run_oos_backtest() in src.backtest.simulator.
    oos_predictions_path: Optional[str] = None

    def to_json(self) -> str:
        payload = {
            "horizon_s": self.horizon_s,
            "interval_ms": self.interval_ms,
            "exchange": self.exchange,
            "symbol": self.symbol,
            "feature_columns": self.feature_columns,
            "final_model_path": self.final_model_path,
            "oos_predictions_path": self.oos_predictions_path,
            "folds": [
                {**asdict(f),
                 "train_start": f.train_start.isoformat(),
                 "train_end": f.train_end.isoformat(),
                 "val_start": f.val_start.isoformat(),
                 "val_end": f.val_end.isoformat()}
                for f in self.folds
            ],
        }
        return json.dumps(payload, indent=2, default=str)


def _load_xy(
    data_root: str | Path, exchange: str, symbol: str,
    dates: List[datetime], interval_ms: int, horizon_s: int,
) -> pd.DataFrame:
    """Load and join features+labels for a list of dates."""
    feat_frames = []
    label_frames = []
    for d in dates:
        ft = read_parquet_dir(features_path(data_root, exchange, symbol, d, interval_ms))
        lt = read_parquet_dir(labels_path(data_root, exchange, symbol, d, interval_ms, horizon_s))
        if ft.num_rows:
            feat_frames.append(ft.to_pandas())
        if lt.num_rows:
            label_frames.append(lt.to_pandas())
    if not feat_frames or not label_frames:
        return pd.DataFrame()
    feats = pd.concat(feat_frames, ignore_index=True).sort_values("ts")
    labels = pd.concat(label_frames, ignore_index=True).sort_values("ts")
    feats["ts"] = pd.to_datetime(feats["ts"], utc=True)
    labels["ts"] = pd.to_datetime(labels["ts"], utc=True)
    df = feats.merge(
        labels[["ts", "exchange", "symbol", "horizon_s", "future_return", "label", "label_class", "threshold_bps"]],
        on=["ts", "exchange", "symbol"], how="inner",
    )
    # Drop invalid (corrupt-book) rows from training.
    if "is_valid" in df.columns:
        df = df[df["is_valid"]].copy()
    df = df.dropna(subset=["mid_price"])
    return df.reset_index(drop=True)


def _feature_columns(df: pd.DataFrame) -> List[str]:
    cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    # Drop string columns (exchange/symbol).
    return [c for c in cols if pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c])]


def train_walk_forward(
    data_root: str | Path,
    exchange: str,
    symbol: str,
    dates: List[datetime],
    interval_ms: int,
    horizon_s: int,
    train_days: int,
    val_days: int,
    step_days: int,
    xgb_params: Dict[str, Any],
    early_stopping_rounds: int = 25,
    out_models_dir: str | Path = "data/models/xgboost",
) -> TrainingResult:
    """
    Run walk-forward training. Returns a TrainingResult with one fold per
    sliding window. Saves the final-fold model to `out_models_dir`.
    """
    df = _load_xy(data_root, exchange, symbol, dates, interval_ms, horizon_s)
    if df.empty:
        logger.warning(f"train: empty dataset for horizon={horizon_s}s")
        return TrainingResult(horizon_s=horizon_s, interval_ms=interval_ms,
                              exchange=exchange, symbol=symbol, feature_columns=[])

    assert_monotonic_time(df, time_col="ts")
    feature_cols = _feature_columns(df)
    logger.info(f"train: {len(df)} rows, {len(feature_cols)} features, horizon={horizon_s}s")

    result = TrainingResult(
        horizon_s=horizon_s, interval_ms=interval_ms,
        exchange=exchange, symbol=symbol, feature_columns=feature_cols,
    )

    one_day = pd.Timedelta(days=1)
    start = df["ts"].min().floor("D")
    end = df["ts"].max().ceil("D")
    cur = start
    fold_idx = 0
    last_booster: Optional[xgb.Booster] = None
    oos_rows: List[pd.DataFrame] = []

    while cur + (train_days + val_days) * one_day <= end:
        train_end = cur + train_days * one_day
        val_end = train_end + val_days * one_day
        train_df = df[(df["ts"] >= cur) & (df["ts"] < train_end)]
        val_df = df[(df["ts"] >= train_end) & (df["ts"] < val_end)]
        if len(train_df) < 1000 or len(val_df) < 100:
            cur = cur + step_days * one_day
            continue

        assert_train_before_val(train_df, val_df, time_col="ts")

        # No StandardScaler: XGBoost is invariant to monotonic feature
        # transformations. Scaling would require we save+load the scaler at
        # inference time, which is one more thing that can silently drift.
        X_train = train_df[feature_cols].astype(float).fillna(0.0).values
        X_val = val_df[feature_cols].astype(float).fillna(0.0).values
        y_train = train_df["label_class"].astype(int).values
        y_val = val_df["label_class"].astype(int).values

        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_cols)
        dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_cols)

        # Adapt sklearn-style params to xgb.train API.
        params = {
            "objective": xgb_params.get("objective", "multi:softprob"),
            "num_class": xgb_params.get("num_class", 3),
            "max_depth": xgb_params["params"].get("max_depth", 6),
            "eta": xgb_params["params"].get("learning_rate", 0.05),
            "subsample": xgb_params["params"].get("subsample", 0.8),
            "colsample_bytree": xgb_params["params"].get("colsample_bytree", 0.8),
            "min_child_weight": xgb_params["params"].get("min_child_weight", 5),
            "reg_alpha": xgb_params["params"].get("reg_alpha", 0.1),
            "reg_lambda": xgb_params["params"].get("reg_lambda", 1.0),
            "tree_method": xgb_params["params"].get("tree_method", "hist"),
            "eval_metric": xgb_params["params"].get("eval_metric", ["mlogloss", "merror"]),
            "verbosity": 0,
        }
        n_rounds = xgb_params["params"].get("n_estimators", 400)
        booster = xgb.train(
            params, dtrain, num_boost_round=n_rounds,
            evals=[(dtrain, "train"), (dval, "val")],
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=False,
        )

        proba = booster.predict(dval)
        pred = np.argmax(proba, axis=1)

        # Capture per-row OOS predictions for downstream backtesting.
        oos_rows.append(pd.DataFrame({
            "ts": val_df["ts"].values,
            "fold_index": fold_idx,
            "mid_price": val_df["mid_price"].values if "mid_price" in val_df.columns else np.nan,
            "spread": val_df["spread"].values if "spread" in val_df.columns else np.nan,
            "is_valid": val_df["is_valid"].values if "is_valid" in val_df.columns else True,
            "prob_short": proba[:, 0],
            "prob_flat":  proba[:, 1],
            "prob_long":  proba[:, 2],
            "y_true_class": y_val,
        }))

        # Per-class precision/recall: class 2 = +1 (long), class 0 = -1 (short)
        acc = float(accuracy_score(y_val, pred))
        ll = float(log_loss(y_val, proba, labels=[0, 1, 2]))
        prec_long = float(precision_score(y_val, pred, labels=[2], average="macro", zero_division=0))
        prec_short = float(precision_score(y_val, pred, labels=[0], average="macro", zero_division=0))
        rec_long = float(recall_score(y_val, pred, labels=[2], average="macro", zero_division=0))
        rec_short = float(recall_score(y_val, pred, labels=[0], average="macro", zero_division=0))
        f1_long = float(f1_score(y_val, pred, labels=[2], average="macro", zero_division=0))
        f1_short = float(f1_score(y_val, pred, labels=[0], average="macro", zero_division=0))
        cm = confusion_matrix(y_val, pred, labels=[0, 1, 2]).tolist()
        imp = booster.get_score(importance_type="gain")
        # Pad missing features with 0.
        imp_full = {c: float(imp.get(c, 0.0)) for c in feature_cols}

        result.folds.append(FoldResult(
            fold_index=fold_idx,
            train_start=train_df["ts"].min(), train_end=train_df["ts"].max(),
            val_start=val_df["ts"].min(), val_end=val_df["ts"].max(),
            n_train=len(train_df), n_val=len(val_df),
            accuracy=acc, log_loss=ll,
            precision_long=prec_long, precision_short=prec_short,
            recall_long=rec_long, recall_short=rec_short,
            f1_long=f1_long, f1_short=f1_short,
            n_signals_long=int((pred == 2).sum()),
            n_signals_short=int((pred == 0).sum()),
            confusion=cm,
            feature_importance_gain=imp_full,
        ))
        logger.info(
            f"fold {fold_idx}: train={len(train_df)} val={len(val_df)} "
            f"acc={acc:.3f} ll={ll:.3f} "
            f"p_long={prec_long:.3f} p_short={prec_short:.3f}"
        )
        last_booster = booster
        fold_idx += 1
        cur = cur + step_days * one_day

    if last_booster is not None:
        out_dir = Path(out_models_dir) / f"horizon_{horizon_s}s"
        out_dir.mkdir(parents=True, exist_ok=True)
        model_path = out_dir / "model.ubj"
        last_booster.save_model(str(model_path))
        result.final_model_path = str(model_path)

        # Save per-fold OOS predictions. The backtester uses these as the
        # source-of-truth out-of-sample stream — no in-sample contamination
        # is possible because every row here came from a fold's validation
        # window, not its training window.
        if oos_rows:
            oos_df = pd.concat(oos_rows, ignore_index=True).sort_values("ts").reset_index(drop=True)
            oos_path = out_dir / "oos_predictions.parquet"
            oos_df.to_parquet(oos_path, index=False)
            result.oos_predictions_path = str(oos_path)
            logger.info(f"train: saved {len(oos_df)} OOS predictions -> {oos_path}")

        # Also save the feature list and a metadata file.
        with open(out_dir / "metadata.json", "w") as f:
            f.write(result.to_json())
        logger.info(f"train: saved final model -> {model_path}")

    return result
