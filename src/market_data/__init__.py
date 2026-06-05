"""Cost-aware public market-data collectors.

This package intentionally uses only public/unauthenticated endpoints. It is a
sidecar to the L3 recorder: it enriches research datasets with derivatives and
market context without introducing CryptoQuant/CoinGlass dependencies.
"""
from __future__ import annotations

from src.market_data.collector import collect_market_data
from src.market_data.records import MarketMetricRecord

__all__ = ["MarketMetricRecord", "collect_market_data"]
