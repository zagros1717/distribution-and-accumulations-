"""
Daily report generator.

Produces a Markdown report with three sections:

  1. Data quality   — message counts, gaps, reconnects, corrupted periods
  2. Model          — windows, metrics, top features, simulated PnL
  3. Decision       — accept / reject the model, with explicit reasons

Markdown is chosen over HTML because it's easy to diff in git, view in any
editor, and render in any chat / wiki tool.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.backtest.simulator import BacktestResult
from src.models.train_xgboost import TrainingResult
from src.storage.parquet_store import raw_path, normalized_path, read_parquet_dir
from src.utils.logging import logger


def _aggregate_feature_importance(folds) -> List[tuple]:
    """Average feature-importance gain across folds."""
    if not folds:
        return []
    keys = folds[0].feature_importance_gain.keys()
    agg = {k: float(np.mean([f.feature_importance_gain.get(k, 0.0) for f in folds])) for k in keys}
    return sorted(agg.items(), key=lambda kv: kv[1], reverse=True)


def _data_quality_section(
    data_root: str | Path, exchange: str, symbol: str, date: datetime
) -> Dict[str, Any]:
    raw_dir = raw_path(data_root, exchange, symbol, date)
    norm_dir = normalized_path(data_root, exchange, symbol, date)
    raw_tbl = read_parquet_dir(raw_dir)
    norm_tbl = read_parquet_dir(norm_dir)
    n_raw = raw_tbl.num_rows
    n_norm = norm_tbl.num_rows

    # Approximate "missing periods" from gaps in receive_time > 5s in normalized.
    missing_periods = 0
    if norm_tbl.num_rows:
        ts = pd.to_datetime(norm_tbl.column("receive_time").to_pandas(), utc=True).sort_values()
        gaps = ts.diff().dt.total_seconds()
        missing_periods = int((gaps > 5).sum())

    # Corrupted periods from snapshots (is_valid=False)
    corrupted_pct = None
    try:
        from src.storage.parquet_store import snapshots_path
        snap_dir = snapshots_path(data_root, exchange, symbol, date, 1000)
        st = read_parquet_dir(snap_dir)
        if st.num_rows:
            df = st.to_pandas()
            corrupted_pct = float((~df["is_valid"]).mean() * 100.0)
    except Exception:
        corrupted_pct = None

    return {
        "raw_messages": int(n_raw),
        "normalized_events": int(n_norm),
        "missing_periods_gt_5s": missing_periods,
        "corrupted_book_pct": corrupted_pct,
    }


def _decision(
    folds, backtest: Optional[BacktestResult], reject_cfg: Dict[str, Any], data_quality: Dict[str, Any]
) -> Dict[str, Any]:
    reasons: List[str] = []
    if not folds:
        reasons.append("no folds trained")
    elif reject_cfg.get("works_on_only_one_day", True) and len(folds) == 1:
        reasons.append("model only validated on a single fold — insufficient evidence")
    if backtest is not None:
        net_pnl = backtest.summary.get("net_pnl_usd", 0.0)
        if reject_cfg.get("pnl_after_costs_negative", True) and net_pnl < 0:
            reasons.append(f"net PnL after costs is negative: {net_pnl:.2f} USD")
        max_dd = backtest.summary.get("max_drawdown_bps", 0.0)
        thr = reject_cfg.get("max_drawdown_bps_above", 500)
        if max_dd > thr:
            reasons.append(f"max drawdown {max_dd:.0f} bps exceeds threshold {thr}")
    if data_quality.get("corrupted_book_pct") is not None:
        cb = data_quality["corrupted_book_pct"]
        thr = reject_cfg.get("corrupted_book_pct_above", 5.0)
        if cb > thr:
            reasons.append(f"corrupted-book percentage {cb:.2f}% exceeds threshold {thr}%")

    accepted = len(reasons) == 0
    return {"accepted": accepted, "reasons": reasons}


def write_daily_report(
    out_dir: str | Path,
    exchange: str,
    symbol: str,
    date: datetime,
    data_root: str | Path,
    training: TrainingResult,
    backtest: Optional[BacktestResult] = None,
    reject_cfg: Optional[Dict[str, Any]] = None,
    top_n_features: int = 30,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    reject_cfg = reject_cfg or {}

    dq = _data_quality_section(data_root, exchange, symbol, date)
    top_features = _aggregate_feature_importance(training.folds)[:top_n_features]
    decision = _decision(training.folds, backtest, reject_cfg, dq)

    lines: List[str] = []
    lines.append(f"# BTC research report — {exchange}/{symbol} — {date.date()}")
    lines.append("")
    lines.append(f"_Generated {datetime.utcnow().isoformat()}Z. **This system performs no trading.**_")
    lines.append("")

    # ---- Data quality ---------------------------------------------------
    lines.append("## 1. Data quality")
    lines.append("")
    lines.append(f"- Raw messages stored: **{dq['raw_messages']:,}**")
    lines.append(f"- Normalized events:   **{dq['normalized_events']:,}**")
    lines.append(f"- Suspected gaps (>5s): **{dq['missing_periods_gt_5s']}**")
    if dq["corrupted_book_pct"] is not None:
        lines.append(f"- Corrupted-book seconds: **{dq['corrupted_book_pct']:.2f}%** of snapshots")
    else:
        lines.append("- Corrupted-book seconds: _no snapshot file_")
    lines.append("")

    # ---- Model ---------------------------------------------------------
    lines.append("## 2. Model")
    lines.append("")
    lines.append(f"- Horizon: **{training.horizon_s}s**")
    lines.append(f"- Feature interval: **{training.interval_ms}ms**")
    lines.append(f"- Feature count: **{len(training.feature_columns)}**")
    lines.append(f"- Walk-forward folds: **{len(training.folds)}**")
    lines.append(f"- Final model: `{training.final_model_path}`")
    lines.append("")
    if training.folds:
        lines.append("### Fold metrics")
        lines.append("")
        lines.append("| Fold | Train | Val | n_train | n_val | Acc | LogLoss | P(long) | P(short) | Signals L/S |")
        lines.append("|---:|---|---|---:|---:|---:|---:|---:|---:|---|")
        for f in training.folds:
            lines.append(
                f"| {f.fold_index} "
                f"| {f.train_start.date()}–{f.train_end.date()} "
                f"| {f.val_start.date()}–{f.val_end.date()} "
                f"| {f.n_train:,} | {f.n_val:,} "
                f"| {f.accuracy:.3f} | {f.log_loss:.3f} "
                f"| {f.precision_long:.3f} | {f.precision_short:.3f} "
                f"| {f.n_signals_long}/{f.n_signals_short} |"
            )
        lines.append("")

        # Confusion matrix of last fold
        last = training.folds[-1]
        lines.append("### Last-fold confusion matrix (rows=true {-1,0,+1}, cols=pred)")
        lines.append("")
        lines.append("| | -1 | 0 | +1 |")
        lines.append("|---|---:|---:|---:|")
        for i, name in enumerate(["-1", "0", "+1"]):
            lines.append(f"| **{name}** | {last.confusion[i][0]} | {last.confusion[i][1]} | {last.confusion[i][2]} |")
        lines.append("")

        # Top features
        lines.append(f"### Top {top_n_features} features by mean gain (across folds)")
        lines.append("")
        lines.append("| # | Feature | Mean gain |")
        lines.append("|---:|---|---:|")
        for i, (name, gain) in enumerate(top_features, 1):
            lines.append(f"| {i} | `{name}` | {gain:.4f} |")
        lines.append("")
    else:
        lines.append("_No folds — training did not produce any windows._")
        lines.append("")

    # ---- Backtest ------------------------------------------------------
    lines.append("## 3. Simulated trading (offline)")
    lines.append("")
    if backtest is None:
        lines.append("_No backtest result attached._")
        lines.append("")
    else:
        s = backtest.summary
        lines.append(f"- Trades simulated:  **{s.get('n_trades', 0)}** ({s.get('n_long', 0)} long, {s.get('n_short', 0)} short)")
        lines.append(f"- Net PnL after costs: **${s.get('net_pnl_usd', 0):,.2f}**")
        lines.append(f"- Gross PnL: ${s.get('gross_pnl_usd', 0):,.2f}, Fees: ${s.get('fees_usd', 0):,.2f}")
        lines.append(f"- Win rate: {s.get('win_rate', 0)*100:.1f}%")
        lines.append(f"- Avg return per trade: {s.get('avg_return_bps', 0):.2f} bps")
        if "max_drawdown_bps" in s:
            lines.append(f"- Max drawdown: ${s['max_drawdown_usd']:.2f} ({s['max_drawdown_bps']:.0f} bps of starting equity)")
        if "sharpe_approx" in s:
            lines.append(f"- Sharpe (approx, annualized): {s['sharpe_approx']:.2f}")
        lines.append("")
        if not backtest.daily_pnl.empty:
            lines.append("### Daily PnL")
            lines.append("")
            lines.append("| Day | Trades | Net PnL | Gross PnL | Fees | Avg bps |")
            lines.append("|---|---:|---:|---:|---:|---:|")
            for _, r in backtest.daily_pnl.iterrows():
                lines.append(
                    f"| {pd.to_datetime(r['day']).date()} | {int(r['n_trades'])} "
                    f"| ${r['net_pnl']:.2f} | ${r['gross_pnl']:.2f} "
                    f"| ${r['fees']:.2f} | {r['avg_bps']:.2f} |"
                )
            lines.append("")

    # ---- Decision ------------------------------------------------------
    lines.append("## 4. Decision")
    lines.append("")
    verdict = "✅ **ACCEPTED**" if decision["accepted"] else "❌ **REJECTED**"
    lines.append(f"{verdict}")
    lines.append("")
    if decision["reasons"]:
        lines.append("Reasons:")
        for r in decision["reasons"]:
            lines.append(f"- {r}")
    else:
        lines.append("No automatic rejection criteria triggered. Manual review still required before any deployment — and this project has no deployment path by design.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("> **Reminder.** This codebase is a research environment. It does not place orders, does not "
                 "connect to private APIs, and the safety guard refuses to load if `execution_enabled` is ever set to true.")

    out_path = out_dir / f"{date.strftime('%Y-%m-%d')}_{exchange}_{symbol}_horizon{training.horizon_s}s.md"
    out_path.write_text("\n".join(lines))
    logger.info(f"report: wrote {out_path}")
    return out_path
