"""
Train a LightGBM classifier on backtested trades to predict P(profitable).
Feature pipeline mirrors what's available in the live executor at signal time.
Output is calibrated via isotonic regression and exported as JSON model + meta.

The executor reads this model URL from BotConfig.ml.model_url and gates signals
through it: trade only if calibrated P(profitable) >= cfg.ml.min_p_profitable.
"""
import argparse, json
from pathlib import Path
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, brier_score_loss


FEATURE_COLS = [
    # If your backtester emits these features per trade, they get used. Otherwise, expand engine.py to record them.
    "atr_pct", "adx_15m", "chop_15m", "htf_stack",
    "vwap_dev_sigma", "funding_bps", "book_imb",
    "cvd_div_sign", "cvd_div_strength", "agg_flip",
    "btc_lead", "realized_vol",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trades", required=True, help="Parquet of backtested trades with feature columns")
    p.add_argument("--out", required=True, help="Output JSON model path")
    p.add_argument("--label", default="r_multiple", help="Column to threshold for the label")
    p.add_argument("--label-threshold", type=float, default=0.0)
    args = p.parse_args()

    df = pd.read_parquet(args.trades)
    if df.empty:
        print("No trades to train on."); return

    feats = [c for c in FEATURE_COLS if c in df.columns]
    if len(feats) < 4:
        print(f"Need at least 4 features in trades parquet; got {feats}.")
        print("Extend research/backtester/engine.py to log these features per trade.")
        return

    X = df[feats].fillna(0).values
    y = (df[args.label] > args.label_threshold).astype(int).values

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    model = lgb.LGBMClassifier(
        n_estimators=400, max_depth=-1, num_leaves=31,
        learning_rate=0.05, min_data_in_leaf=20, reg_lambda=1.0,
        objective="binary", n_jobs=-1,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], callbacks=[lgb.early_stopping(20)])

    p_te = model.predict_proba(X_te)[:, 1]
    auc = roc_auc_score(y_te, p_te); brier = brier_score_loss(y_te, p_te)

    # Isotonic calibration on the holdout
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_te, y_te)
    p_te_cal = iso.predict(p_te)

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(out.with_suffix(".lgb")))
    meta = {
        "features": feats,
        "auc": float(auc), "brier": float(brier),
        "calibration_x": iso.X_thresholds_.tolist(),
        "calibration_y": iso.y_thresholds_.tolist(),
        "label": args.label, "label_threshold": args.label_threshold,
        "n_train": int(len(X_tr)), "n_test": int(len(X_te)),
    }
    out.write_text(json.dumps(meta, indent=2))
    print(f"AUC={auc:.3f} Brier={brier:.4f}")
    print(f"Saved model to {out.with_suffix('.lgb')} and meta to {out}")


if __name__ == "__main__":
    main()
