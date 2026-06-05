from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Calibration:
    metric_key: str
    n_samples: int
    hit_rate_long: float
    hit_rate_short: float
    avg_return_when_high: float
    avg_return_when_low: float
    reliability: float


def _safe_corr(a: pd.Series, b: pd.Series) -> float:
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return 0.0
    val = float(a.corr(b))
    return 0.0 if np.isnan(val) else val


def calibrate_history(history: pd.DataFrame, min_samples: int = 50) -> dict[str, Calibration]:
    """Calibrate metric usefulness from point-in-time history.

    Expected columns: metric_key, value, forward_return_24h.
    Returns conservative reliability values in [0, 1].
    """
    if history.empty:
        return {}
    out: dict[str, Calibration] = {}
    df = history.copy()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["forward_return_24h"] = pd.to_numeric(df["forward_return_24h"], errors="coerce")
    df = df.dropna(subset=["metric_key", "value", "forward_return_24h"])

    for key, g in df.groupby("metric_key"):
        g = g.sort_index()
        n = len(g)
        if n < min_samples:
            out[str(key)] = Calibration(str(key), n, 0.0, 0.0, 0.0, 0.0, 0.0)
            continue
        q_low = g["value"].quantile(0.2)
        q_high = g["value"].quantile(0.8)
        high = g[g["value"] >= q_high]
        low = g[g["value"] <= q_low]
        avg_high = float(high["forward_return_24h"].mean()) if not high.empty else 0.0
        avg_low = float(low["forward_return_24h"].mean()) if not low.empty else 0.0
        hit_long = float((high["forward_return_24h"] > 0).mean()) if not high.empty else 0.0
        hit_short = float((low["forward_return_24h"] < 0).mean()) if not low.empty else 0.0
        corr = abs(_safe_corr(g["value"], g["forward_return_24h"]))
        edge = min(abs(avg_high - avg_low) * 100.0, 1.0)  # returns are decimal; cap aggressively
        reliability = max(0.0, min(1.0, 0.5 * corr + 0.5 * edge))
        out[str(key)] = Calibration(str(key), n, hit_long, hit_short, avg_high, avg_low, reliability)
    return out


def load_calibration_csv(path: str | None, min_samples: int = 50) -> dict[str, Calibration]:
    if not path:
        return {}
    df = pd.read_csv(path)
    return calibrate_history(df, min_samples=min_samples)
