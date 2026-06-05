"""Public, unauthenticated market-data providers for research context."""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any, Iterable, Mapping, Sequence

import aiohttp

from src.market_data.http_cache import HTTPCache
from src.market_data.records import MarketMetricRecord, from_unix_millis, parse_float, utcnow


class MarketDataProvider(ABC):
    name: str

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = dict(config or {})

    @abstractmethod
    async def collect(self, session: aiohttp.ClientSession, cache: HTTPCache) -> list[MarketMetricRecord]:
        pass

    def _symbols(self, default: Sequence[str] = ("BTCUSDT",)) -> list[str]:
        symbols = self.config.get("symbols", default)
        return [symbols] if isinstance(symbols, str) else [str(s) for s in symbols]

    def _ttl(self, metric: str, default: int) -> int:
        ttl = self.config.get("ttl_seconds", {})
        if isinstance(ttl, Mapping):
            return int(ttl.get(metric, default))
        return int(ttl or default)


class BinanceFuturesProvider(MarketDataProvider):
    name = "binance_futures"
    base_url = "https://fapi.binance.com"

    async def collect(self, session: aiohttp.ClientSession, cache: HTTPCache) -> list[MarketMetricRecord]:
        out: list[MarketMetricRecord] = []
        metrics = set(self.config.get("metrics", ["open_interest", "funding_rate", "global_long_short_ratio"]))
        period = self.config.get("period", "5m")
        limit = int(self.config.get("limit", 30))
        for symbol in self._symbols():
            if "open_interest" in metrics:
                payload = await cache.get_json(session, f"{self.base_url}/fapi/v1/openInterest", params={"symbol": symbol}, ttl_seconds=self._ttl("open_interest", 60))
                out.append(MarketMetricRecord(self.name, "open_interest", symbol, from_unix_millis(payload.get("time")), parse_float(payload.get("openInterest")), "contracts", payload))
            if "funding_rate" in metrics:
                rows = await cache.get_json(session, f"{self.base_url}/fapi/v1/fundingRate", params={"symbol": symbol, "limit": limit}, ttl_seconds=self._ttl("funding_rate", 300))
                out.extend(MarketMetricRecord(self.name, "funding_rate", str(r.get("symbol", symbol)), from_unix_millis(r.get("fundingTime")), parse_float(r.get("fundingRate")), "rate", r) for r in rows)
            if "global_long_short_ratio" in metrics:
                rows = await cache.get_json(session, f"{self.base_url}/futures/data/globalLongShortAccountRatio", params={"symbol": symbol, "period": period, "limit": limit}, ttl_seconds=self._ttl("global_long_short_ratio", 300))
                out.extend(MarketMetricRecord(self.name, "global_long_short_ratio", str(r.get("symbol", symbol)), from_unix_millis(r.get("timestamp")), parse_float(r.get("longShortRatio")), "ratio", r) for r in rows)
        return out


class BybitProvider(MarketDataProvider):
    name = "bybit"
    base_url = "https://api.bybit.com"

    async def collect(self, session: aiohttp.ClientSession, cache: HTTPCache) -> list[MarketMetricRecord]:
        out: list[MarketMetricRecord] = []
        metrics = set(self.config.get("metrics", ["open_interest", "funding_rate", "long_short_ratio"]))
        category = self.config.get("category", "linear")
        interval = self.config.get("interval", "5min")
        period = self.config.get("period", "5min")
        limit = int(self.config.get("limit", 50))
        for symbol in self._symbols():
            if "open_interest" in metrics:
                payload = await cache.get_json(session, f"{self.base_url}/v5/market/open-interest", params={"category": category, "symbol": symbol, "intervalTime": interval, "limit": limit}, ttl_seconds=self._ttl("open_interest", 300))
                out.extend(MarketMetricRecord(self.name, "open_interest", symbol, from_unix_millis(r.get("timestamp")), parse_float(r.get("openInterest")), "contracts_or_coin", r) for r in payload.get("result", {}).get("list", []))
            if "funding_rate" in metrics:
                payload = await cache.get_json(session, f"{self.base_url}/v5/market/funding/history", params={"category": category, "symbol": symbol, "limit": limit}, ttl_seconds=self._ttl("funding_rate", 300))
                out.extend(MarketMetricRecord(self.name, "funding_rate", str(r.get("symbol", symbol)), from_unix_millis(r.get("fundingRateTimestamp")), parse_float(r.get("fundingRate")), "rate", r) for r in payload.get("result", {}).get("list", []))
            if "long_short_ratio" in metrics:
                payload = await cache.get_json(session, f"{self.base_url}/v5/market/account-ratio", params={"category": category, "symbol": symbol, "period": period, "limit": limit}, ttl_seconds=self._ttl("long_short_ratio", 300))
                for r in payload.get("result", {}).get("list", []):
                    buy = parse_float(r.get("buyRatio")); sell = parse_float(r.get("sellRatio"))
                    ratio = buy / sell if buy is not None and sell not in (None, 0.0) else None
                    out.append(MarketMetricRecord(self.name, "long_short_ratio", str(r.get("symbol", symbol)), from_unix_millis(r.get("timestamp")), ratio, "ratio", r))
        return out


class CoinGeckoProvider(MarketDataProvider):
    name = "coingecko"
    base_url = "https://api.coingecko.com/api/v3"

    async def collect(self, session: aiohttp.ClientSession, cache: HTTPCache) -> list[MarketMetricRecord]:
        coin_ids = self.config.get("coin_ids", ["bitcoin"])
        coin_ids = [coin_ids] if isinstance(coin_ids, str) else coin_ids
        headers = {"x-cg-demo-api-key": os.getenv("COINGECKO_API_KEY")} if os.getenv("COINGECKO_API_KEY") else None
        out: list[MarketMetricRecord] = []
        for coin_id in coin_ids:
            payload = await cache.get_json(session, f"{self.base_url}/coins/{coin_id}/market_chart", params={"vs_currency": self.config.get("vs_currency", "usd"), "days": str(self.config.get("days", "1")), "interval": self.config.get("interval")}, headers=headers, ttl_seconds=self._ttl("market_chart", 60))
            out.extend(self._chart_records(str(coin_id), payload))
        return out

    def _chart_records(self, coin_id: str, payload: Mapping[str, Any]) -> list[MarketMetricRecord]:
        mapping = {"prices": ("price", "usd"), "market_caps": ("market_cap", "usd"), "total_volumes": ("volume", "usd")}
        out: list[MarketMetricRecord] = []
        for key, (metric, unit) in mapping.items():
            for ts_ms, value in payload.get(key, []) or []:
                out.append(MarketMetricRecord(self.name, metric, coin_id, from_unix_millis(ts_ms), parse_float(value), unit, {"source_key": key, "value": value}))
        return out


class DefiLlamaProvider(MarketDataProvider):
    name = "defillama"
    chain_tvl_url = "https://api.llama.fi/v2/chains"
    stablecoins_url = "https://stablecoins.llama.fi/stablecoins"

    async def collect(self, session: aiohttp.ClientSession, cache: HTTPCache) -> list[MarketMetricRecord]:
        metrics = set(self.config.get("metrics", ["chain_tvl", "stablecoins_mcap"]))
        out: list[MarketMetricRecord] = []
        if "chain_tvl" in metrics:
            rows = await cache.get_json(session, self.chain_tvl_url, ttl_seconds=self._ttl("chain_tvl", 3600))
            now = utcnow()
            out.extend(MarketMetricRecord(self.name, "chain_tvl", str(r.get("name", r.get("gecko_id", "unknown"))), now, parse_float(r.get("tvl")), "usd", r) for r in (rows if isinstance(rows, list) else []))
        if "stablecoins_mcap" in metrics:
            payload = await cache.get_json(session, self.stablecoins_url, params={"includePrices": "true"}, ttl_seconds=self._ttl("stablecoins_mcap", 3600))
            now = utcnow()
            for r in payload.get("peggedAssets", []) if isinstance(payload, Mapping) else []:
                circ = r.get("circulating") if isinstance(r, Mapping) else None
                mcap = circ.get("peggedUSD") if isinstance(circ, Mapping) else None
                out.append(MarketMetricRecord(self.name, "stablecoin_mcap", str(r.get("symbol", r.get("name", "unknown"))), now, parse_float(mcap), "usd", r))
        return out


PROVIDERS: dict[str, type[MarketDataProvider]] = {
    BinanceFuturesProvider.name: BinanceFuturesProvider,
    BybitProvider.name: BybitProvider,
    CoinGeckoProvider.name: CoinGeckoProvider,
    DefiLlamaProvider.name: DefiLlamaProvider,
}
