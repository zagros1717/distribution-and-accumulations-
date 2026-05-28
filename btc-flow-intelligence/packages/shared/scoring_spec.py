"""
Single source of truth for the scoring model.

This module is intentionally dependency-free so it can be imported by the
backend, used in tests, or transpiled/mirrored to the frontend. The TypeScript
mirror lives in apps/frontend/lib/api.ts and MUST be kept in sync with the
values here. A unit test (test_scoring.py) asserts the weights sum to 1.0.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Verdict(str, Enum):
    ACCUMULATION = "Accumulation"
    DISTRIBUTION = "Distribution"
    NEUTRAL = "Mixed/Neutral"


class Confidence(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


@dataclass(frozen=True)
class Category:
    key: str          # stable machine key, used as DB `category`
    label: str        # human label for the UI
    weight: float     # fraction of the final weighted score


# Order here defines display order in the Signal Matrix.
CATEGORIES: list[Category] = [
    Category("etf_flows", "ETF Flows", 0.20),
    Category("onchain_flows", "Exchange / On-chain Flows", 0.20),
    Category("derivatives", "Derivatives Positioning", 0.20),
    Category("entity_flows", "Whale / Miner / Entity Flows", 0.15),
    Category("market_structure", "Spot / Futures Liquidity", 0.10),
    Category("stablecoin", "Stablecoin Liquidity", 0.10),
    Category("sentiment", "Sentiment / Valuation", 0.05),
]

CATEGORY_BY_KEY: dict[str, Category] = {c.key: c for c in CATEGORIES}

# Discrete per-signal score scale.
SCORE_SCALE = {
    2: "Strong accumulation",
    1: "Mild accumulation",
    0: "Neutral",
    -1: "Mild distribution",
    -2: "Strong distribution",
}
SCORE_MIN, SCORE_MAX = -2, 2

# Verdict thresholds applied to the final weighted score (range -2..+2).
ACCUMULATION_THRESHOLD = 0.50
DISTRIBUTION_THRESHOLD = -0.50


def classify(weighted_score: float) -> Verdict:
    if weighted_score > ACCUMULATION_THRESHOLD:
        return Verdict.ACCUMULATION
    if weighted_score < DISTRIBUTION_THRESHOLD:
        return Verdict.DISTRIBUTION
    return Verdict.NEUTRAL


def assert_weights_valid() -> None:
    total = round(sum(c.weight for c in CATEGORIES), 6)
    if total != 1.0:
        raise ValueError(f"Category weights must sum to 1.0, got {total}")


assert_weights_valid()
