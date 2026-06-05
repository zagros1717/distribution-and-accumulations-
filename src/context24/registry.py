from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Category = Literal["market_structure", "etf_flows", "derivatives", "entity_flows", "sentiment", "market_data"]
DirectionRule = Literal[
    "positive_is_bullish",
    "negative_is_bullish",
    "high_is_bearish",
    "low_is_bullish",
    "neutral_only",
]


@dataclass(frozen=True)
class MetricDefinition:
    key: str
    source: str
    metric: str
    category: Category
    unit: str
    freshness_minutes: int
    direction_rule: DirectionRule
    requires_delta: bool = False
    min_history_days: int = 180
    min_samples: int = 50
    reliability_weight: float = 1.0


def metric_key(source: str, metric: str) -> str:
    return f"{source.strip().lower()}.{metric.strip().lower().replace(' ', '_')}"


_DEF = [
    MetricDefinition("coingecko.btc_price", "coingecko", "btc price", "market_data", "usd", 15, "neutral_only", reliability_weight=0.7),
    MetricDefinition("coinbase.orderbook_imbalance", "coinbase", "orderbook imbalance", "market_structure", "z", 5, "positive_is_bullish"),
    MetricDefinition("coinbase.recent_trade_imbalance", "coinbase", "recent trade imbalance", "market_structure", "z", 5, "positive_is_bullish"),
    MetricDefinition("farside.etf_net_flow_usd_m", "farside", "etf net flow usd m", "etf_flows", "usd_m", 1440, "positive_is_bullish", reliability_weight=0.8),
    MetricDefinition("sosovalue.etf_holdings_change_pct", "sosovalue", "etf holdings change pct", "etf_flows", "pct", 1440, "positive_is_bullish", reliability_weight=0.8),
    MetricDefinition("coinglass.open_interest", "coinglass", "open interest", "derivatives", "usd_b", 30, "negative_is_bullish", requires_delta=True, reliability_weight=0.7),
    MetricDefinition("coinglass.funding_rate", "coinglass", "funding rate", "derivatives", "pct", 30, "high_is_bearish", reliability_weight=0.7),
    MetricDefinition("coinglass.long_short_ratio", "coinglass", "long short ratio", "derivatives", "ratio", 30, "high_is_bearish", reliability_weight=0.7),
    MetricDefinition("coinglass.liquidation_skew", "coinglass", "liquidation skew", "derivatives", "pct", 30, "negative_is_bullish", reliability_weight=0.7),
    MetricDefinition("deribit.options_put_call_ratio", "deribit", "options put call ratio", "derivatives", "ratio", 60, "low_is_bullish", reliability_weight=0.8),
    MetricDefinition("cme.cme_open_interest", "cme", "cme open interest", "derivatives", "contracts", 1440, "negative_is_bullish", requires_delta=True, reliability_weight=0.8),
    MetricDefinition("cme.cme_annualised_basis", "cme", "cme annualised basis", "derivatives", "pct", 1440, "high_is_bearish", reliability_weight=0.7),
    MetricDefinition("arkham.whale_net_exchange_flow", "arkham", "whale net exchange flow", "entity_flows", "btc", 120, "negative_is_bullish", reliability_weight=0.5),
    MetricDefinition("arkham.etf_custody_net_change", "arkham", "etf custody net change", "entity_flows", "btc", 120, "positive_is_bullish", reliability_weight=0.5),
    MetricDefinition("arkham.government_wallet_change", "arkham", "government wallet change", "entity_flows", "btc", 1440, "negative_is_bullish", reliability_weight=0.4),
    MetricDefinition("arkham.treasury_wallet_change", "arkham", "treasury wallet change", "entity_flows", "btc", 1440, "positive_is_bullish", reliability_weight=0.4),
    MetricDefinition("kaiko.spot_volume", "kaiko", "spot volume", "market_structure", "usd_b", 60, "neutral_only", reliability_weight=0.6),
    MetricDefinition("kaiko.futures_volume", "kaiko", "futures volume", "market_structure", "usd_b", 60, "neutral_only", reliability_weight=0.6),
    MetricDefinition("kaiko.spot_futures_ratio", "kaiko", "spot futures ratio", "market_structure", "ratio", 60, "positive_is_bullish", reliability_weight=0.6),
    MetricDefinition("kaiko.orderbook_imbalance", "kaiko", "orderbook imbalance", "market_structure", "z", 60, "positive_is_bullish", reliability_weight=0.6),
    MetricDefinition("fear_greed.fear_greed_index", "fear_greed", "fear greed index", "sentiment", "index", 1440, "low_is_bullish", reliability_weight=0.5),
]

METRICS = {m.key: m for m in _DEF}

CATEGORY_CAPS = {
    "market_structure": 0.25,
    "etf_flows": 0.20,
    "derivatives": 0.25,
    "entity_flows": 0.15,
    "sentiment": 0.15,
    "market_data": 0.05,
}
