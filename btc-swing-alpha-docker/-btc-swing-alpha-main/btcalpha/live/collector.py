"""
btcalpha.live.collector
~~~~~~~~~~~~~~~~~~~~~~~
Background collector for fetching real candles, building features/labels,
warming models/cache, and saving lightweight artifacts without waiting for a dashboard click.

Storage policy for Railway free volumes:
- Always writes latest snapshot only.
- Archive snapshots are disabled by default.
- artifact_mode="minimal" stores candles + labels + metadata only.
"""
from __future__ import annotations

import asyncio
import json
import shutil
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Optional

import pandas as pd

from btcalpha.config import get_config, get_logger, resolve_path
from btcalpha.live.engine import PipelineSnapshot, run_pipeline

log = get_logger("collector")

COLLECTOR_STATE: Dict[str, dict] = {
    "enabled": False,
    "started_at": None,
    "max_concurrent": 1,
    "timeframes": {},
}
_TASKS: list[asyncio.Task] = []
_SEMAPHORE: asyncio.Semaphore | None = None


def _json_default(obj):
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    if hasattr(obj, "item"):
        return obj.item()
    return str(obj)


def _write_df(df: pd.DataFrame | pd.Series, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(df, pd.Series):
        df = df.to_frame(name=df.name or "value")
    try:
        df.to_parquet(path.with_suffix(".parquet"))
        return str(path.with_suffix(".parquet"))
    except Exception:  # noqa: BLE001
        df.to_pickle(path.with_suffix(".pkl"))
        return str(path.with_suffix(".pkl"))


def _dir_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total


def _write_snapshot_files(snap: PipelineSnapshot, base: Path, cfg: dict, artifact_mode: str) -> dict:
    """Write snapshot artifacts.

    minimal: candles + labels + metadata only.
    standard: minimal + decisions + regime.
    full: all intermediate frames.
    metadata_only: metadata JSON only.
    """
    artifacts = {}

    if artifact_mode in {"minimal", "standard", "full"}:
        artifacts["candles"] = _write_df(snap.dataset, base / "candles")
        artifacts["labels"] = _write_df(snap.features["y"], base / "labels")

    if artifact_mode in {"standard", "full"}:
        artifacts["decisions"] = _write_df(snap.decisions, base / "decisions")
        artifacts["regime"] = _write_df(snap.regime, base / "regime")

    if artifact_mode == "full":
        artifacts["features"] = _write_df(snap.features["X"], base / "features")
        artifacts["raw"] = _write_df(snap.features["raw"], base / "raw")
        artifacts["probabilities"] = _write_df(snap.proba, base / "probabilities")

    meta = {
        "timeframe": snap.timeframe,
        "generated_at": snap.generated_at,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "artifact_mode": artifact_mode,
        "source_exchanges": cfg["data"]["exchanges"],
        "primary_exchange": cfg["data"].get("primary_exchange"),
        "synthetic_fallback": bool(cfg["data"].get("allow_synthetic_fallback", False)),
        "model_type": cfg["model"]["type"],
        "rows_dataset": int(len(snap.dataset)),
        "rows_features": int(len(snap.features["X"])),
        "rows_labeled": int(snap.features["y"].notna().sum()),
        "first_candle": str(snap.dataset.index[0]) if len(snap.dataset) else None,
        "last_candle": str(snap.dataset.index[-1]) if len(snap.dataset) else None,
        "last_price": float(snap.dataset["close"].iloc[-1]) if len(snap.dataset) else None,
        "feature_count": int(len(snap.features["feature_cols"])),
        "model_meta": asdict(snap.model.meta) if snap.model.meta else None,
        "live": snap.live,
        "backtest_metrics": snap.backtest.metrics,
    }
    meta_path = base / "metadata.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    artifacts["metadata"] = str(meta_path)
    return artifacts


def _prune_archives(tf_dir: Path, keep: int) -> int:
    archive_root = tf_dir / "archive"
    if not archive_root.exists() or keep < 0:
        return 0
    archives = sorted([p for p in archive_root.iterdir() if p.is_dir()], key=lambda p: p.name, reverse=True)
    removed = 0
    for old in archives[keep:]:
        shutil.rmtree(old, ignore_errors=True)
        removed += 1
    return removed


def save_snapshot(snap: PipelineSnapshot) -> dict:
    cfg = get_config()
    collector_cfg = cfg.get("collector", {})
    archive_enabled = bool(collector_cfg.get("archive_enabled", False))
    max_archives = int(collector_cfg.get("max_archives_per_timeframe", 0))
    artifact_mode = str(collector_cfg.get("artifact_mode", "minimal")).lower()
    if artifact_mode not in {"metadata_only", "minimal", "standard", "full"}:
        artifact_mode = "minimal"

    save_dir = resolve_path(collector_cfg.get("save_dir", "storage/collector"))
    tf_dir = save_dir / snap.timeframe
    latest_dir = tf_dir / "latest"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive_dir = tf_dir / "archive" / stamp

    artifacts = {str(latest_dir): _write_snapshot_files(snap, latest_dir, cfg, artifact_mode)}
    actual_archive_dir = None

    if archive_enabled:
        artifacts[str(archive_dir)] = _write_snapshot_files(snap, archive_dir, cfg, artifact_mode)
        actual_archive_dir = str(archive_dir)

    removed_archives = _prune_archives(tf_dir, max_archives)

    return {
        "save_dir": str(save_dir),
        "latest_dir": str(latest_dir),
        "archive_dir": actual_archive_dir,
        "archive_enabled": archive_enabled,
        "artifact_mode": artifact_mode,
        "removed_archives": removed_archives,
        "latest_size_bytes": _dir_size_bytes(latest_dir),
        "artifacts": artifacts,
    }


def collect_once(timeframe: str, force_retrain: Optional[bool] = None) -> PipelineSnapshot:
    cfg = get_config()
    collector_cfg = cfg.get("collector", {})
    if force_retrain is None:
        force_retrain = bool(collector_cfg.get("force_retrain", False))

    t0 = time.time()
    COLLECTOR_STATE.setdefault("timeframes", {}).setdefault(timeframe, {})
    COLLECTOR_STATE["timeframes"][timeframe].update({
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "error": None,
    })

    snap = run_pipeline(timeframe, force_retrain=force_retrain)
    saved = save_snapshot(snap)
    elapsed = round(time.time() - t0, 2)

    COLLECTOR_STATE["timeframes"][timeframe].update({
        "status": "ok",
        "last_run_at": datetime.now(timezone.utc).isoformat(),
        "last_elapsed_seconds": elapsed,
        "last_candle": str(snap.dataset.index[-1]) if len(snap.dataset) else None,
        "last_price": float(snap.dataset["close"].iloc[-1]) if len(snap.dataset) else None,
        "rows_dataset": int(len(snap.dataset)),
        "rows_labeled": int(snap.features["y"].notna().sum()),
        "model_type": cfg["model"]["type"],
        "latest_dir": saved["latest_dir"],
        "archive_dir": saved["archive_dir"],
        "archive_enabled": saved["archive_enabled"],
        "artifact_mode": saved["artifact_mode"],
        "removed_archives": saved["removed_archives"],
        "latest_size_bytes": saved["latest_size_bytes"],
        "error": None,
    })
    log.info(
        "collector %s OK in %.2fs | candle=%s | artifact_mode=%s | archive=%s",
        timeframe,
        elapsed,
        COLLECTOR_STATE["timeframes"][timeframe]["last_candle"],
        saved["artifact_mode"],
        saved["archive_enabled"],
    )
    return snap


async def _collector_loop(timeframe: str, interval: int, initial_delay: int, on_snapshot: Optional[Callable[[str, PipelineSnapshot], None]] = None):
    await asyncio.sleep(max(0, initial_delay))
    global _SEMAPHORE
    while True:
        try:
            sem = _SEMAPHORE or asyncio.Semaphore(1)
            COLLECTOR_STATE.setdefault("timeframes", {}).setdefault(timeframe, {})["waiting_for_slot"] = True
            async with sem:
                COLLECTOR_STATE["timeframes"][timeframe]["waiting_for_slot"] = False
                snap = await asyncio.to_thread(collect_once, timeframe, None)
            if on_snapshot:
                on_snapshot(timeframe, snap)
        except Exception as exc:  # noqa: BLE001
            log.error("collector %s failed: %s", timeframe, exc)
            COLLECTOR_STATE.setdefault("timeframes", {}).setdefault(timeframe, {}).update({
                "status": "error",
                "last_error_at": datetime.now(timezone.utc).isoformat(),
                "error": str(exc),
            })
        await asyncio.sleep(interval)


def start_background_collector(on_snapshot: Optional[Callable[[str, PipelineSnapshot], None]] = None) -> dict:
    cfg = get_config()
    collector_cfg = cfg.get("collector", {})
    enabled = bool(collector_cfg.get("enabled", False))
    max_concurrent = max(1, int(collector_cfg.get("max_concurrent", 1)))
    global _SEMAPHORE
    _SEMAPHORE = asyncio.Semaphore(max_concurrent)

    COLLECTOR_STATE["enabled"] = enabled
    COLLECTOR_STATE["started_at"] = datetime.now(timezone.utc).isoformat()
    COLLECTOR_STATE["max_concurrent"] = max_concurrent
    COLLECTOR_STATE["archive_enabled"] = bool(collector_cfg.get("archive_enabled", False))
    COLLECTOR_STATE["max_archives_per_timeframe"] = int(collector_cfg.get("max_archives_per_timeframe", 0))
    COLLECTOR_STATE["artifact_mode"] = str(collector_cfg.get("artifact_mode", "minimal")).lower()

    if not enabled:
        log.info("collector disabled by config")
        return COLLECTOR_STATE

    intervals = collector_cfg.get("intervals_seconds", {})
    default_interval = int(collector_cfg.get("default_interval_seconds", 900))
    startup_gap = int(collector_cfg.get("startup_stagger_seconds", 90))
    timeframes = collector_cfg.get("timeframes") or cfg["data"]["timeframes"]

    for i, tf in enumerate(timeframes):
        interval = int(intervals.get(tf, default_interval))
        initial_delay = i * startup_gap
        COLLECTOR_STATE.setdefault("timeframes", {}).setdefault(tf, {
            "status": "scheduled",
            "interval_seconds": interval,
            "initial_delay_seconds": initial_delay,
        })
        task = asyncio.create_task(_collector_loop(tf, interval, initial_delay, on_snapshot))
        _TASKS.append(task)
        log.info("collector scheduled: %s every %ss after %ss", tf, interval, initial_delay)

    return COLLECTOR_STATE


def get_collector_status() -> dict:
    return COLLECTOR_STATE
