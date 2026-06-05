#!/usr/bin/env python3
"""Collect cheap public market/context metrics.

Usage:
    python scripts/collect_market_data.py --config config/config.yaml

This command reads the `market_data` section from the normal project config and
writes JSONL partitions under data/market_metrics/.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter

from src.market_data.collector import collect_market_data
from src.utils.config import load_config
from src.utils.logging import setup_logging


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect public market metrics without CryptoQuant/CoinGlass.")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
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
