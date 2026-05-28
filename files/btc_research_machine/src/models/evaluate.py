"""Helpers to load a saved XGBoost model and emit predictions.

NOTE: This codebase deliberately does NOT use feature scaling. XGBoost is
invariant to monotonic feature transformations, so a StandardScaler would add
no model quality while creating a save/load consistency hazard. If you ever
add a scaling step at training time, you MUST persist the fitted scaler here
and load it during inference, or predictions will be garbage.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb


def load_model(model_dir: str | Path) -> Tuple[xgb.Booster, List[str]]:
    model_dir = Path(model_dir)
    booster = xgb.Booster()
    booster.load_model(str(model_dir / "model.ubj"))
    with open(model_dir / "metadata.json") as f:
        meta = json.load(f)
    return booster, meta["feature_columns"]


def predict_proba(
    booster: xgb.Booster,
    feature_cols: List[str],
    df: pd.DataFrame,
) -> np.ndarray:
    """Return P(class) per row, with class order [-1, 0, +1] => columns 0,1,2."""
    X = df[feature_cols].astype(float).fillna(0.0).values
    dm = xgb.DMatrix(X, feature_names=feature_cols)
    return booster.predict(dm)


def classify(proba: np.ndarray, min_confidence: float = 0.0) -> np.ndarray:
    """
    Return -1/0/+1 class. If max prob across classes < min_confidence,
    output 0 (no-signal).
    """
    pred = np.argmax(proba, axis=1) - 1  # 0,1,2 -> -1,0,+1
    if min_confidence > 0:
        conf = proba.max(axis=1)
        pred = np.where(conf >= min_confidence, pred, 0)
    return pred
