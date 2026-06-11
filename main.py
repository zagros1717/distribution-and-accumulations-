"""
Main entrypoint.

Subcommands:

  record        Live recorder. Primary Bitfinex, fallback Coinbase. No execution.
  replay        Reconstruct order book from normalized parquet, emit snapshots.
  features      Build feature parquet for a (date, interval_ms).
  labels        Build labels for a (date, horizon_s).
  train         Walk-forward XGBoost training across a date range.
  backtest      Run offline simulator using a trained model.
  report        Produce the daily Markdown report.
  pipeline      Run replay + features + labels + train + report end-to-end.
  onchain-flow  Pull Arkham BTC transfers and report large exchange inflow/outflow.

All commands first call assert_research_mode() via load_config().
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from src.utils.config import load_config
from src.utils.logging import setup_logging, logger


def _parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def _daterange(start: datetime, end: datetime) -> List[datetime]:
    out = []
    d = start
    while d <= end:
        out.append(d)
        d += timedelta(days=1)
    return out


def _backtest_config_from_cfg(cfg: dict):
    """Build BacktestConfig while preserving dataclass defaults for new knobs."""
    from src.backtest.simulator import BacktestConfig
    bt = cfg.get("backtest", {})
    kwargs = {
        name: bt[name]
        for name in BacktestConfig.__dataclass_fields__.keys()
        if name in bt
    }
    return BacktestConfig(**kwargs)


# ----- subcommand handlers ----------------------------------------------------

def cmd_record(args, cfg) -> int:
    from src.recorder import main_async
    asyncio.run(main_async(cfg))
    return 0


def cmd_replay(args, cfg) -> int:
    from src.book.reconstructor import replay_day
    exchange = args.exchange
    symbol = args.symbol
    n = 0
    for d in _daterange(_parse_date(args.start), _parse_date(args.end)):
        for ival in cfg["storage"]["snapshot_intervals_ms"]:
            n += replay_day(cfg["storage"]["root"], exchange, symbol, d, interval_ms=ival)
    logger.info(f"replay: total snapshots written = {n}")
    return 0


def cmd_features(args, cfg) -> int:
    from src.features.feature_engine import generate_features
    for d in _daterange(_parse_date(args.start), _parse_date(args.end)):
        for ival in cfg["features"]["intervals_ms"]:
            generate_features(
                cfg["storage"]["root"], args.exchange, args.symbol, d,
                interval_ms=ival,
                depth_levels=cfg["features"]["depth_levels"],
                large_order_threshold_btc=cfg["features"]["large_order_threshold_btc"],
            )
    return 0


def cmd_labels(args, cfg) -> int:
    from src.labels.label_engine import generate_labels
    for d in _daterange(_parse_date(args.start), _parse_date(args.end)):
        for horizon in cfg["labels"]["horizons_seconds"]:
            for ival in cfg["features"]["intervals_ms"]:
                generate_labels(
                    cfg["storage"]["root"], args.exchange, args.symbol, d,
                    interval_ms=ival, horizon_s=horizon,
                    cost_components_bps=cfg["labels"]["cost_components_bps"],
                )
    return 0


def cmd_train(args, cfg) -> int:
    from src.models.train_xgboost import train_walk_forward
    dates = _daterange(_parse_date(args.start), _parse_date(args.end))
    wf = cfg["model"]["walk_forward"]
    result = train_walk_forward(
        data_root=cfg["storage"]["root"],
        exchange=args.exchange, symbol=args.symbol,
        dates=dates,
        interval_ms=cfg["features"]["intervals_ms"][0],
        horizon_s=args.horizon,
        train_days=wf["train_days"], val_days=wf["validation_days"],
        step_days=wf["step_days"],
        xgb_params={"objective": cfg["model"]["objective"],
                     "num_class": cfg["model"]["num_class"],
                     "params": cfg["model"]["params"]},
        early_stopping_rounds=cfg["model"]["early_stopping_rounds"],
        out_models_dir=Path(cfg["storage"]["root"]) / "models" / "xgboost",
    )
    print(result.to_json())
    return 1 if result.rejection_reasons else 0


def cmd_backtest(args, cfg) -> int:
    """Out-of-sample backtest using validation-window predictions only."""
    from src.backtest.simulator import run_oos_backtest
    from src.storage.parquet_store import features_path, read_parquet_dir
    from src.utils.validation import assert_monotonic_time

    model_dir = Path(cfg["storage"]["root"]) / "models" / "xgboost" / f"horizon_{args.horizon}s"
    oos_path = model_dir / "oos_predictions.parquet"
    if not oos_path.exists():
        logger.error(
            f"backtest: no OOS predictions at {oos_path}. "
            f"Run `train` first; OOS backtests cannot use raw predictions over "
            f"the training range (that would be in-sample)."
        )
        return 1

    feats = []
    for d in _daterange(_parse_date(args.start), _parse_date(args.end)):
        t = read_parquet_dir(features_path(
            cfg["storage"]["root"], args.exchange, args.symbol, d,
            cfg["features"]["intervals_ms"][0],
        ))
        if t.num_rows:
            feats.append(t.to_pandas())
    if not feats:
        logger.error("backtest: no feature data in date range")
        return 1
    df = pd.concat(feats, ignore_index=True).sort_values("ts").reset_index(drop=True)
    assert_monotonic_time(df)

    result = run_oos_backtest(oos_path, df, horizon_s=args.horizon, config=_backtest_config_from_cfg(cfg))
    print(json.dumps(result.summary, indent=2, default=str))
    return 0


def cmd_report(args, cfg) -> int:
    """Train, run an OOS backtest, and write the report."""
    from src.models.train_xgboost import train_walk_forward
    from src.reports.daily_report import write_daily_report
    from src.backtest.simulator import run_oos_backtest
    from src.storage.parquet_store import features_path, read_parquet_dir

    dates = _daterange(_parse_date(args.start), _parse_date(args.end))
    wf = cfg["model"]["walk_forward"]
    training = train_walk_forward(
        data_root=cfg["storage"]["root"],
        exchange=args.exchange, symbol=args.symbol,
        dates=dates,
        interval_ms=cfg["features"]["intervals_ms"][0],
        horizon_s=args.horizon,
        train_days=max(wf["min_train_days"], 1),
        val_days=wf["validation_days"],
        step_days=wf["step_days"],
        xgb_params={"objective": cfg["model"]["objective"],
                     "num_class": cfg["model"]["num_class"],
                     "params": cfg["model"]["params"]},
        early_stopping_rounds=cfg["model"]["early_stopping_rounds"],
        out_models_dir=Path(cfg["storage"]["root"]) / "models" / "xgboost",
    )
    bt_result = None
    if training.oos_predictions_path:
        feats = []
        for d in dates:
            t = read_parquet_dir(features_path(
                cfg["storage"]["root"], args.exchange, args.symbol, d,
                cfg["features"]["intervals_ms"][0],
            ))
            if t.num_rows:
                feats.append(t.to_pandas())
        if feats:
            df = pd.concat(feats, ignore_index=True).sort_values("ts").reset_index(drop=True)
            bt_result = run_oos_backtest(
                training.oos_predictions_path, df,
                horizon_s=args.horizon, config=_backtest_config_from_cfg(cfg),
            )

    write_daily_report(
        out_dir=cfg["reports"]["output_dir"],
        exchange=args.exchange, symbol=args.symbol,
        date=dates[-1],
        data_root=cfg["storage"]["root"],
        training=training,
        backtest=bt_result,
        reject_cfg=cfg["reports"]["reject_if"],
        top_n_features=cfg["reports"]["top_n_features"],
    )
    return 1 if training.rejection_reasons else 0


def cmd_pipeline(args, cfg) -> int:
    """Run the full offline pipeline for a date range."""
    rc = cmd_replay(args, cfg)
    if rc != 0: return rc
    rc = cmd_features(args, cfg)
    if rc != 0: return rc
    rc = cmd_labels(args, cfg)
    if rc != 0: return rc
    rc = cmd_train(args, cfg)
    if rc != 0: return rc
    rc = cmd_report(args, cfg)
    return rc


def cmd_onchain_flow(args, cfg) -> int:
    """Pull read-only Arkham BTC transfer data and write an exchange-flow report."""
    from src.onchain.arkham_flow import run_arkham_flow_report

    summary = run_arkham_flow_report(
        cfg=cfg,
        start=args.start,
        end=args.end,
        dry_run=args.dry_run,
    )
    print(json.dumps(summary.as_dict(), indent=2))
    return 0


# ----- arg parsing ------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="btc_research_machine",
                                     description="Offline Bitcoin order-book research (no trading).")
    parser.add_argument("--config", default="config/config.yaml")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("record", help="Live recorder (Bitfinex primary, Coinbase fallback)")

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--exchange", required=True, choices=["bitfinex", "coinbase"])
    common.add_argument("--symbol", required=True)
    common.add_argument("--start", required=True, help="YYYY-MM-DD")
    common.add_argument("--end", required=True, help="YYYY-MM-DD")

    sub.add_parser("replay", parents=[common], help="Reconstruct book + snapshots")
    sub.add_parser("features", parents=[common], help="Generate features parquet")
    sub.add_parser("labels", parents=[common], help="Generate labels parquet")

    train = sub.add_parser("train", parents=[common], help="Walk-forward XGBoost training")
    train.add_argument("--horizon", type=int, required=True)

    bt = sub.add_parser("backtest", parents=[common], help="Run offline backtest")
    bt.add_argument("--horizon", type=int, required=True)

    rep = sub.add_parser("report", parents=[common], help="Train + backtest + write report")
    rep.add_argument("--horizon", type=int, required=True)

    pl = sub.add_parser("pipeline", parents=[common], help="replay -> features -> labels -> train -> report")
    pl.add_argument("--horizon", type=int, required=True)

    flow = sub.add_parser(
        "onchain-flow",
        help="Read Arkham BTC transfers and report large-wallet exchange inflow/outflow",
    )
    flow.add_argument("--start", default=None, help="UTC start: YYYY-MM-DD or ISO timestamp. Default: now-24h")
    flow.add_argument("--end", default=None, help="UTC end: YYYY-MM-DD or ISO timestamp. Default: now")
    flow.add_argument("--dry-run", action="store_true", help="Write an empty report without calling Arkham")

    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    setup_logging(level=cfg["logging"]["level"],
                  sink=cfg["logging"].get("sink"),
                  json_logs=cfg["logging"].get("json_logs", False))

    dispatch = {
        "record": cmd_record,
        "replay": cmd_replay,
        "features": cmd_features,
        "labels": cmd_labels,
        "train": cmd_train,
        "backtest": cmd_backtest,
        "report": cmd_report,
        "pipeline": cmd_pipeline,
        "onchain-flow": cmd_onchain_flow,
    }
    return dispatch[args.cmd](args, cfg)


if __name__ == "__main__":
    sys.exit(main())
