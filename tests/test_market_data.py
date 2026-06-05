from __future__ import annotations

from datetime import timezone

import pytest

from src.market_data.providers import BinanceFuturesProvider, BybitProvider, CoinGeckoProvider
from src.market_data.records import from_unix_millis, parse_float


class FakeCache:
    def __init__(self, payloads):
        self.payloads = payloads
        self.calls = []

    async def get_json(self, session, url, *, params=None, headers=None, ttl_seconds=None):
        self.calls.append({"url": url, "params": params, "headers": headers, "ttl_seconds": ttl_seconds})
        for key, payload in self.payloads.items():
            if key in url:
                return payload
        raise AssertionError(f"unexpected url {url}")


def test_parse_helpers_are_safe():
    assert parse_float("1.25") == 1.25
    assert parse_float(2) == 2.0
    assert parse_float("") is None
    assert parse_float("not-a-number") is None
    assert from_unix_millis("0").tzinfo == timezone.utc


@pytest.mark.asyncio
async def test_binance_provider_normalizes_public_derivatives_metrics():
    cache = FakeCache({
        "openInterest": {"symbol": "BTCUSDT", "openInterest": "123.45", "time": 1000},
        "fundingRate": [{"symbol": "BTCUSDT", "fundingRate": "0.0001", "fundingTime": 2000}],
        "globalLongShortAccountRatio": [{"symbol": "BTCUSDT", "longShortRatio": "1.5", "timestamp": "3000"}],
    })
    provider = BinanceFuturesProvider({"symbols": ["BTCUSDT"], "limit": 1})

    records = await provider.collect(None, cache)

    assert [r.metric for r in records] == ["open_interest", "funding_rate", "global_long_short_ratio"]
    assert records[0].value == 123.45
    assert records[1].value == 0.0001
    assert records[2].value == 1.5
    assert all("cryptoquant" not in c["url"].lower() and "coinglass" not in c["url"].lower() for c in cache.calls)


@pytest.mark.asyncio
async def test_bybit_provider_derives_long_short_ratio():
    cache = FakeCache({
        "open-interest": {"result": {"list": [{"openInterest": "10", "timestamp": "1000"}]}},
        "funding/history": {"result": {"list": [{"symbol": "BTCUSDT", "fundingRate": "0.01", "fundingRateTimestamp": "2000"}]}},
        "account-ratio": {"result": {"list": [{"symbol": "BTCUSDT", "buyRatio": "0.60", "sellRatio": "0.40", "timestamp": "3000"}]}}
    })
    provider = BybitProvider({"symbols": ["BTCUSDT"], "limit": 1})

    records = await provider.collect(None, cache)

    assert [r.metric for r in records] == ["open_interest", "funding_rate", "long_short_ratio"]
    assert records[-1].value == 1.5


def test_coingecko_market_chart_normalizes_price_cap_volume():
    provider = CoinGeckoProvider({})
    records = provider._chart_records("bitcoin", {
        "prices": [[1000, 50000]],
        "market_caps": [[1000, 900000000000]],
        "total_volumes": [[1000, 10000000000]],
    })

    assert [r.metric for r in records] == ["price", "market_cap", "volume"]
    assert [r.value for r in records] == [50000.0, 900000000000.0, 10000000000.0]
