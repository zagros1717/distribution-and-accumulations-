"""
btcalpha.model.alpha_model
~~~~~~~~~~~~~~~~~~~~~~~~~~
Alpha ML model.

Important trading note:
  Model probabilities are later used by the strategy as p_up/p_down/p_neutral
  for thresholds and Kelly sizing. Because of that, class-balancing weights must
  be optional. Weighted classification can improve class recall but often makes
  predict_proba poorly calibrated. The default is no class weighting.

Split discipline:
  Trade-outcome labels look forward by `horizon` bars. Therefore the production
  train/valid/test split purges `horizon` bars before each boundary so labels in
  train do not depend on validation prices, and validation labels do not depend
  on test prices.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

from btcalpha.config import get_config, get_logger, resolve_path

log = get_logger("model")


@dataclass
class ModelMeta:
    timeframe: str
    model_type: str
    feature_cols: list
    trained_at: str
    n_train: int
    n_valid: int
    n_test: int
    valid_accuracy: float
    test_accuracy: float
    class_distribution: dict
    horizon: int
    majority_class: int = -1
    baseline_valid_accuracy: float = 0.0
    baseline_test_accuracy: float = 0.0
    accuracy_lift_vs_baseline: float = 0.0
    train_class_distribution: dict | None = None
    valid_class_distribution: dict | None = None
    test_class_distribution: dict | None = None
    data_start: str | None = None
    data_end: str | None = None
    fit_start: str | None = None
    fit_end: str | None = None
    valid_start: str | None = None
    valid_end: str | None = None
    test_start: str | None = None
    test_end: str | None = None
    class_weighting: str | None = None
    label_mode: str | None = None
    embargo_bars: int = 0
    split_method: str = "chronological"


class AlphaModel:
    """One model per timeframe: train, save/load and predict."""

    def __init__(self, timeframe: str):
        self.timeframe = timeframe
        self.cfg = get_config()
        self.model = None
        self.meta: Optional[ModelMeta] = None
        self.feature_cols: list = []
        self._scaler_mean = None
        self._scaler_std = None

    def _paths(self):
        mdir = resolve_path(self.cfg["model"]["model_dir"])
        tag = self.timeframe.replace("/", "")
        return {
            "model": mdir / f"alpha_{tag}.pkl",
            "meta": mdir / f"alpha_{tag}_meta.json",
        }

    def _horizon(self) -> int:
        return int(self.cfg.get("features", {}).get("horizons", {}).get(self.timeframe, 0) or 0)

    def _time_split(self, X: pd.DataFrame, y: pd.Series):
        """Chronological split with label-horizon purge before boundaries.

        If the raw 70/15/15 boundaries are tr and va, the rows [tr-h, tr) and
        [va-h, va) are discarded. This prevents samples immediately before a
        boundary from using future prices inside the next split through their
        forward-looking labels.
        """
        n = len(X)
        h = max(0, self._horizon())
        tr = int(n * self.cfg["model"]["train_ratio"])
        va = int(n * (self.cfg["model"]["train_ratio"] + self.cfg["model"]["valid_ratio"]))

        train_end = max(0, tr - h)
        valid_start = min(n, tr)
        valid_end = max(valid_start, va - h)
        test_start = min(n, va)

        Xtr, ytr = X.iloc[:train_end], y.iloc[:train_end]
        Xva, yva = X.iloc[valid_start:valid_end], y.iloc[valid_start:valid_end]
        Xte, yte = X.iloc[test_start:], y.iloc[test_start:]

        if len(Xtr) < 200 or len(Xva) < 20 or len(Xte) < 20:
            log.warning(
                "purged split is small — train:%d valid:%d test:%d embargo:%d raw_n:%d",
                len(Xtr), len(Xva), len(Xte), h, n,
            )
        return (Xtr, ytr), (Xva, yva), (Xte, yte)

    @staticmethod
    def _range_start_end(X: pd.DataFrame) -> tuple[str | None, str | None]:
        if X is None or len(X) == 0:
            return None, None
        return str(X.index[0]), str(X.index[-1])

    def _fit_scaler(self, X: pd.DataFrame):
        self._scaler_mean = X.mean()
        self._scaler_std = X.std().replace(0, 1.0)

    def _scale(self, X: pd.DataFrame) -> pd.DataFrame:
        return (X - self._scaler_mean) / self._scaler_std

    def _class_weighting_mode(self) -> str:
        return str(self.cfg.get("model", {}).get("class_weighting", "none")).strip().lower()

    def _sample_weight(self, y: pd.Series):
        mode = self._class_weighting_mode()
        if mode in {"none", "off", "false", "0", "no"}:
            return None
        if mode not in {"balanced", "inverse_frequency", "auto"}:
            log.warning("class_weighting=%s ناشناخته است؛ بدون وزن‌دهی آموزش می‌دهیم.", mode)
            return None
        classes, counts = np.unique(y.astype(int), return_counts=True)
        if len(classes) < 2:
            return None
        cw = {c: len(y) / (len(classes) * n) for c, n in zip(classes, counts)}
        return y.astype(int).map(cw).values

    @staticmethod
    def _class_dist(y: pd.Series) -> dict:
        if y is None or len(y) == 0:
            return {}
        vc = y.astype(int).value_counts().sort_index()
        return {int(k): int(v) for k, v in vc.items()}

    @staticmethod
    def _class_dist_pct(y: pd.Series) -> dict:
        if y is None or len(y) == 0:
            return {}
        vc = y.astype(int).value_counts(normalize=True).sort_index()
        return {int(k): round(float(v), 4) for k, v in vc.items()}

    def _build_estimator(self):
        mtype = self.cfg["model"]["type"]
        if mtype == "lightgbm":
            try:
                import lightgbm as lgb  # noqa
            except (ImportError, OSError) as exc:
                log.warning("lightgbm در این محیط قابل بارگذاری نیست (%s) — به logistic سقوط می‌کنیم.", exc)
                mtype = "logistic"
            else:
                import lightgbm as lgb
                return lgb.LGBMClassifier(**self.cfg["model"]["lightgbm_params"])

        from sklearn.linear_model import LogisticRegression
        cw = "balanced" if self._class_weighting_mode() in {"balanced", "inverse_frequency", "auto"} else None
        return LogisticRegression(max_iter=1000, class_weight=cw)

    def feature_schema_check(self, feat: dict) -> tuple[bool, dict]:
        """Return whether a loaded model matches the currently-built features.

        This protects production from stale model artifacts. For example, when
        macro model features are disabled, old 4h/1h/15m artifacts may still
        contain macro_* columns in their saved feature list. Without this check,
        predict_proba would fail with "[...] not in index". A mismatch means the
        model must be retrained with the current feature schema.
        """
        current_cols = list(feat.get("feature_cols", []))
        saved_cols = list(self.feature_cols or [])
        missing_in_current = [c for c in saved_cols if c not in current_cols]
        new_extra_cols = [c for c in current_cols if c not in saved_cols]
        same_order = saved_cols == current_cols
        ok = same_order and not missing_in_current and not new_extra_cols
        return ok, {
            "saved_feature_count": len(saved_cols),
            "current_feature_count": len(current_cols),
            "missing_in_current": missing_in_current,
            "new_extra_cols": new_extra_cols,
            "same_order": same_order,
        }

    def train(self, feat: dict) -> ModelMeta:
        X_all, y_all = feat["X"], feat["y"]
        mask = y_all.notna()
        X = X_all[mask].copy()
        y = y_all[mask].astype(int).copy()
        self.feature_cols = feat["feature_cols"]

        if len(X) < 200:
            raise ValueError(f"داده برای آموزش کافی نیست: {len(X)} ردیف")

        (Xtr, ytr), (Xva, yva), (Xte, yte) = self._time_split(X, y)
        log.info(
            "تقسیم زمانی با purge — train:%d  valid:%d  test:%d  embargo:%d",
            len(Xtr), len(Xva), len(Xte), self._horizon(),
        )

        if len(Xtr) < 200 or len(Xva) == 0 or len(Xte) == 0:
            raise ValueError(
                f"تقسیم train/valid/test بعد از embargo کافی نیست: train={len(Xtr)} valid={len(Xva)} test={len(Xte)}"
            )

        self._fit_scaler(Xtr)
        Xtr_s, Xva_s, Xte_s = self._scale(Xtr), self._scale(Xva), self._scale(Xte)

        self.model = self._build_estimator()

        classes = np.unique(ytr)
        if len(classes) < 2:
            raise ValueError(f"تنوع کلاس برای آموزش کافی نیست: فقط کلاس {classes.tolist()} در train دیده شد")

        sample_weight = self._sample_weight(ytr)
        if sample_weight is None:
            log.info("آموزش بدون class weighting؛ احتمال‌ها طبیعی‌تر ولی recall کلاس‌های کم‌تعداد ممکن است کمتر شود.")
            try:
                self.model.fit(Xtr_s, ytr, eval_set=[(Xva_s, yva)])
            except TypeError:
                self.model.fit(Xtr_s, ytr)
        else:
            log.info("آموزش با class weighting=%s؛ probability ممکن است کالیبراسیون ضعیف‌تری داشته باشد.", self._class_weighting_mode())
            try:
                self.model.fit(
                    Xtr_s, ytr,
                    sample_weight=sample_weight,
                    eval_set=[(Xva_s, yva)],
                )
            except TypeError:
                self.model.fit(Xtr_s, ytr, sample_weight=sample_weight)

        va_pred = self.model.predict(Xva_s) if len(Xva) else []
        te_pred = self.model.predict(Xte_s) if len(Xte) else []
        va_acc = float((va_pred == yva).mean()) if len(Xva) else 0.0
        te_acc = float((te_pred == yte).mean()) if len(Xte) else 0.0

        majority_class = int(pd.Series(ytr).value_counts().idxmax())
        base_va = float((yva == majority_class).mean()) if len(yva) else 0.0
        base_te = float((yte == majority_class).mean()) if len(yte) else 0.0
        lift = te_acc - base_te

        log.info(
            "دقت — valid: %.3f   test: %.3f   baseline_test: %.3f   lift: %.3f",
            va_acc, te_acc, base_te, lift,
        )

        data_start, data_end = self._range_start_end(X)
        fit_start, fit_end = self._range_start_end(Xtr)
        valid_start, valid_end = self._range_start_end(Xva)
        test_start, test_end = self._range_start_end(Xte)

        train_dist = self._class_dist(ytr)
        self.meta = ModelMeta(
            timeframe=self.timeframe,
            model_type=self.cfg["model"]["type"],
            feature_cols=self.feature_cols,
            trained_at=datetime.now(timezone.utc).isoformat(),
            n_train=len(Xtr),
            n_valid=len(Xva),
            n_test=len(Xte),
            valid_accuracy=va_acc,
            test_accuracy=te_acc,
            class_distribution=train_dist,
            horizon=self._horizon(),
            majority_class=majority_class,
            baseline_valid_accuracy=base_va,
            baseline_test_accuracy=base_te,
            accuracy_lift_vs_baseline=lift,
            train_class_distribution={"count": train_dist, "pct": self._class_dist_pct(ytr)},
            valid_class_distribution={"count": self._class_dist(yva), "pct": self._class_dist_pct(yva)},
            test_class_distribution={"count": self._class_dist(yte), "pct": self._class_dist_pct(yte)},
            data_start=data_start,
            data_end=data_end,
            fit_start=fit_start,
            fit_end=fit_end,
            valid_start=valid_start,
            valid_end=valid_end,
            test_start=test_start,
            test_end=test_end,
            class_weighting=self._class_weighting_mode(),
            label_mode=str(self.cfg.get("features", {}).get("label_mode", "direction")),
            embargo_bars=self._horizon(),
            split_method="chronological_purged_horizon_embargo",
        )
        return self.meta

    def save(self):
        if self.model is None:
            raise RuntimeError("مدلی برای ذخیره وجود ندارد.")
        import pickle

        paths = self._paths()
        bundle = {
            "model": self.model,
            "scaler_mean": self._scaler_mean,
            "scaler_std": self._scaler_std,
            "feature_cols": self.feature_cols,
        }
        with open(paths["model"], "wb") as f:
            pickle.dump(bundle, f)
        with open(paths["meta"], "w", encoding="utf-8") as f:
            json.dump(asdict(self.meta), f, indent=2, ensure_ascii=False)
        log.info("مدل ذخیره شد: %s", paths["model"].name)

    def load(self) -> bool:
        import pickle

        paths = self._paths()
        if not paths["model"].exists():
            log.warning("مدل ذخیره‌شده‌ای برای %s پیدا نشد.", self.timeframe)
            return False
        with open(paths["model"], "rb") as f:
            bundle = pickle.load(f)
        self.model = bundle["model"]
        self._scaler_mean = bundle["scaler_mean"]
        self._scaler_std = bundle["scaler_std"]
        self.feature_cols = bundle["feature_cols"]
        if paths["meta"].exists():
            with open(paths["meta"], "r", encoding="utf-8") as f:
                meta_raw = json.load(f)
                known = {field.name for field in ModelMeta.__dataclass_fields__.values()}
                meta_raw = {k: v for k, v in meta_raw.items() if k in known}
                self.meta = ModelMeta(**meta_raw)
        log.info("مدل %s بارگذاری شد (آموزش: %s).", self.timeframe, self.meta.trained_at if self.meta else "?")
        return True

    def needs_retrain(self) -> bool:
        if not self.cfg["model"]["retrain"]["enabled"]:
            return False
        if self.meta is None:
            return True
        trained = datetime.fromisoformat(self.meta.trained_at)
        age_days = (datetime.now(timezone.utc) - trained).days
        return age_days >= self.cfg["model"]["retrain"]["every_n_days"]

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("ابتدا مدل را train یا load کن.")
        missing = [c for c in self.feature_cols if c not in X.columns]
        if missing:
            raise ValueError(
                f"feature schema mismatch for {self.timeframe}: model expects {len(self.feature_cols)} cols, "
                f"input has {len(X.columns)} cols, missing={missing[:20]}"
            )
        X = X[self.feature_cols]
        Xs = self._scale(X)
        proba = self.model.predict_proba(Xs)
        classes = list(self.model.classes_)
        out = pd.DataFrame(index=X.index)
        out["p_down"] = proba[:, classes.index(0)] if 0 in classes else 0.0
        out["p_neutral"] = proba[:, classes.index(1)] if 1 in classes else 0.0
        out["p_up"] = proba[:, classes.index(2)] if 2 in classes else 0.0
        return out

    def feature_importance(self) -> Optional[pd.Series]:
        if self.model is None:
            return None
        if hasattr(self.model, "feature_importances_"):
            return pd.Series(self.model.feature_importances_, index=self.feature_cols).sort_values(ascending=False)
        if hasattr(self.model, "coef_"):
            imp = np.abs(self.model.coef_).mean(axis=0)
            return pd.Series(imp, index=self.feature_cols).sort_values(ascending=False)
        return None


def get_or_train_model(feat: dict, force_retrain: bool = False) -> AlphaModel:
    tf = feat["timeframe"]
    m = AlphaModel(tf)
    loaded = m.load()
    if loaded:
        schema_ok, schema_audit = m.feature_schema_check(feat)
        if not schema_ok:
            log.warning(
                "مدل ذخیره‌شده‌ی %s با feature schema فعلی سازگار نیست؛ بازآموزی اجباری. audit=%s",
                tf,
                schema_audit,
            )
            force_retrain = True

    if loaded and not force_retrain and not m.needs_retrain():
        return m
    if loaded and m.needs_retrain():
        log.info("مدل %s قدیمی است — بازآموزی ...", tf)
    m.train(feat)
    m.save()
    return m
