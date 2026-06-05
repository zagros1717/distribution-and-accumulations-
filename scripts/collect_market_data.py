#!/usr/bin/env python3
"""Collect cheap public market/context metrics.

Usage:
    python scripts/collect_market_data.py --config config/config.yaml

The command loads the normal project config and, when present, overlays
`config/market_data.yaml` so the L3 pipeline config can stay minimal.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from src.market_data.collector import collect_market_data
from src.utils.config import load_config
from src.utils.logging import setup_logging


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def _load_market_overlay(config_path: str) -> dict[str, Any]:
    cfg_path = Path(config_path)
    candidates = [cfg_path.with_name("market_data.yaml"), Path("config/market_data.yaml")]
    for candidate in candidates:
        if candidate.exists():
            with candidate.open("r") as f:
                return yaml.safe_load(f) or {}
    return {}


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect public market metrics without CryptoQuant/CoinGlass.")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = _deep_merge(cfg, _load_market_overlay(args.config))
    setup_logging(
        level=cfg.get("logging", {}).get("level", "INFO"),
        sink=cfg.get("logging", {}).get("sink"),
        json_logs=cfg.get("logging", {}).get("json_logs", False),
    )

    records = asyncio.run(collect_market_data(cfg))
    counts = Counter((r.source, r.metric) for r in records)
    print(json.dumps({
        "records": len(records),
        "by_source_metric": {f"{src}.{metric}": n for (src, metric), n in sorted(counts.items())},
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
