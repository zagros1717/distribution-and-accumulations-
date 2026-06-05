"""Collector orchestration and local storage for public market metrics."""
from __future__ import annotations

import json
from datetime import timezone
from pathlib import Path
from typing import Any, Mapping

import aiohttp

from src.market_data.http_cache import HTTPCache
from src.market_data.providers import PROVIDERS
from src.market_data.records import MarketMetricRecord
from src.utils.logging import logger


def _metric_dir(root: str | Path, rec: MarketMetricRecord) -> Path:
    date = rec.ts.astimezone(timezone.utc).strftime("%Y-%m-%d")
    safe_symbol = rec.symbol.replace("/", "_").replace(" ", "_")
    return Path(root) / "market_metrics" / f"source={rec.source}" / f"metric={rec.metric}" / f"symbol={safe_symbol}" / f"date={date}"


def write_jsonl(root: str | Path, records: list[MarketMetricRecord]) -> list[Path]:
    """Append records to transparent JSONL partition files."""
    paths: dict[Path, list[dict[str, Any]]] = {}
    for rec in records:
        out_dir = _metric_dir(root, rec)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "metrics.jsonl"
        paths.setdefault(path, []).append(rec.to_dict())

    written: list[Path] = []
    for path, rows in paths.items():
        with path.open("a") as f:
            for row in rows:
                f.write(json.dumps(row, sort_keys=True, default=str) + "\n")
        written.append(path)
    return written


async def collect_market_data(config: Mapping[str, Any]) -> list[MarketMetricRecord]:
    """Collect enabled public market metrics and write them to disk."""
    market_cfg = dict(config.get("market_data", {}) or {})
    if not market_cfg.get("enabled", False):
        logger.warning("market_data: disabled in config")
        return []

    data_root = config.get("storage", {}).get("root", "./data")
    cache_cfg = dict(market_cfg.get("cache", {}) or {})
    cache = HTTPCache(
        cache_cfg.get("dir", str(Path(data_root) / "http_cache")),
        default_ttl_seconds=int(cache_cfg.get("default_ttl_seconds", 300)),
        enabled=bool(cache_cfg.get("enabled", True)),
    )

    records: list[MarketMetricRecord] = []
    providers_cfg = dict(market_cfg.get("providers", {}) or {})
    async with aiohttp.ClientSession(headers={"User-Agent": "btc-research-machine/1.0"}) as session:
        for name, cfg in providers_cfg.items():
            if not cfg or not cfg.get("enabled", False):
                continue
            provider_cls = PROVIDERS.get(name)
            if provider_cls is None:
                logger.warning(f"market_data: unknown provider {name!r}; skipping")
                continue
            provider = provider_cls(cfg)
            try:
                batch = await provider.collect(session, cache)
                records.extend(batch)
                logger.info(f"market_data: {name} collected {len(batch)} records")
            except Exception as e:
                logger.exception(f"market_data: {name} failed: {e}")
                if market_cfg.get("fail_fast", False):
                    raise

    paths = write_jsonl(data_root, records)
    logger.info(f"market_data: wrote {len(records)} records to {len(paths)} partitions")
    return records
