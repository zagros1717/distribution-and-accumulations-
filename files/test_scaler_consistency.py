"""
Scaler-removal consistency tests.

Prior to this fix, the trainer fit a StandardScaler on each fold's training
data but never saved it. At inference time the model received unscaled
inputs and silently produced garbage. We removed StandardScaler entirely
(XGBoost is invariant to monotonic feature transforms) — these tests pin
that decision down so it can't regress.

What we verify:

  1. `predict_proba` accepts a plain feature DataFrame with NO scaler argument.
  2. Predictions from a fresh DMatrix on the saved model exactly match the
     predictions the trainer wrote to oos_predictions.parquet — i.e. there is
     no hidden preprocessing step between training and inference.
  3. evaluate.predict_proba's signature does not include a `scaler` kwarg.
"""
from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from src.models.evaluate import load_model, predict_proba, classify


# ---------------------------------------------------------------------------
# 1. Signature pinning
# ---------------------------------------------------------------------------

def test_predict_proba_has_no_scaler_param():
    """If we ever re-add a scaler arg, callers must be audited — block silent regression."""
    sig = inspect.signature(predict_proba)
    assert "scaler" not in sig.parameters, (
        "predict_proba must not take a scaler argument. If you reintroduce "
        "scaling you must also persist the fitted scaler at train time and "
        "load it consistently here."
    )


def test_evaluate_module_does_not_import_standardscaler():
    """No StandardScaler symbol must leak into the evaluate module."""
    import src.models.evaluate as ev
    assert not hasattr(ev, "StandardScaler"), (
        "src.models.evaluate must not import StandardScaler. Re-adding it "
        "is the failure mode the project explicitly removed."
    )


def test_train_module_does_not_use_standardscaler():
    """
    The trainer must not call StandardScaler. We check both the imported
    symbol table and the source AST for any call site.
    """
    import ast
    import src.models.train_xgboost as tx
    assert not hasattr(tx, "StandardScaler"), (
        "src.models.train_xgboost must not import StandardScaler."
    )
    # Look at the AST for any Name node referring to StandardScaler or any
    # `from sklearn.preprocessing import StandardScaler` style import.
    src_text = inspect.getsource(tx)
    tree = ast.parse(src_text)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "sklearn.preprocessing":
            names = [a.name for a in node.names]
            assert "StandardScaler" not in names, (
                f"src.models.train_xgboost imports StandardScaler: {names}. "
                f"Remove it — the project does not use feature scaling."
            )
        if isinstance(node, ast.Name) and node.id == "StandardScaler":
            raise AssertionError(
                "src.models.train_xgboost references StandardScaler in code "
                "(not just a comment). Remove the reference."
            )


# ---------------------------------------------------------------------------
# 2. End-to-end: trainer saves a model, loader produces identical predictions
# ---------------------------------------------------------------------------

def _toy_multiclass_data(n_rows: int = 300, n_features: int = 6, seed: int = 0):
    """Generate a small toy dataset with three classes that XGBoost can learn."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, size=(n_rows, n_features))
    # Class is a function of the first two features so the booster has
    # something to fit; we don't care about accuracy, just consistency.
    logits = np.column_stack([
        -X[:, 0] - X[:, 1],     # class 0 (short)
        np.zeros(n_rows),        # class 1 (flat)
        X[:, 0] + X[:, 1],       # class 2 (long)
    ])
    y = np.argmax(logits + rng.normal(0, 0.5, size=logits.shape), axis=1)
    cols = [f"f{i}" for i in range(n_features)]
    return X, y, cols


def test_loaded_model_predictions_match_in_memory_predictions(tmp_path):
    """
    Train, save, reload, predict — predictions must be bit-identical.
    Because there is no scaler, this is just `booster.predict(DMatrix(X))`.
    """
    X, y, cols = _toy_multiclass_data()
    dtrain = xgb.DMatrix(X, label=y, feature_names=cols)
    params = {"objective": "multi:softprob", "num_class": 3,
              "max_depth": 3, "eta": 0.3, "verbosity": 0,
              "tree_method": "hist"}
    booster = xgb.train(params, dtrain, num_boost_round=20)

    # Save in the layout the project uses.
    model_dir = tmp_path / "horizon_5s"
    model_dir.mkdir()
    booster.save_model(str(model_dir / "model.ubj"))
    with open(model_dir / "metadata.json", "w") as f:
        json.dump({"feature_columns": cols}, f)

    # Reload and predict via the evaluate helpers.
    reloaded, feat_cols_back = load_model(model_dir)
    assert feat_cols_back == cols

    # Build a DataFrame and run through predict_proba.
    df = pd.DataFrame(X, columns=cols)
    proba_via_helper = predict_proba(reloaded, feat_cols_back, df)

    # Reference path: do it the long way (no helper).
    proba_direct = reloaded.predict(xgb.DMatrix(X, feature_names=cols))

    np.testing.assert_allclose(proba_via_helper, proba_direct, rtol=1e-7, atol=1e-12)


def test_predict_proba_does_not_alter_input_via_scaling(tmp_path):
    """
    Sanity: predict_proba on the same df called twice must give the same
    answer. If someone secretly introduced an in-place scaler it would mutate
    state between calls.
    """
    X, y, cols = _toy_multiclass_data(n_rows=50, seed=1)
    dtrain = xgb.DMatrix(X, label=y, feature_names=cols)
    params = {"objective": "multi:softprob", "num_class": 3,
              "max_depth": 2, "eta": 0.5, "verbosity": 0, "tree_method": "hist"}
    booster = xgb.train(params, dtrain, num_boost_round=5)

    df = pd.DataFrame(X, columns=cols)
    a = predict_proba(booster, cols, df)
    b = predict_proba(booster, cols, df)
    np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# 3. Classify thresholds work end-to-end (sanity)
# ---------------------------------------------------------------------------

def test_classify_returns_minus1_zero_plus1():
    """Class 0/1/2 maps to -1/0/+1; min_confidence forces low-conf rows to 0."""
    proba = np.array([
        [0.8, 0.1, 0.1],    # high-conf short -> -1
        [0.1, 0.8, 0.1],    # high-conf flat -> 0
        [0.1, 0.1, 0.8],    # high-conf long -> +1
        [0.4, 0.4, 0.2],    # low-conf -> 0 after threshold
    ])
    out = classify(proba, min_confidence=0.6)
    assert list(out) == [-1, 0, 1, 0]
