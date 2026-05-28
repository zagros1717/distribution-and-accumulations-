"""Adapter registry. Add a new source here and it joins the pipeline."""

from __future__ import annotations

from app.sources.arkham import ArkhamAdapter
from app.sources.base import SourceAdapter
from app.sources.cme import CMEAdapter
from app.sources.coinglass import CoinGlassAdapter
from app.sources.cryptoquant import CryptoQuantAdapter
from app.sources.deribit import DeribitAdapter
from app.sources.farside import FarsideAdapter
from app.sources.fear_greed import FearGreedAdapter
from app.sources.kaiko import KaikoAdapter
from app.sources.price import PriceAdapter
from app.sources.sosovalue import SoSoValueAdapter


def all_adapters() -> list[SourceAdapter]:
    return [
        PriceAdapter(),
        FarsideAdapter(),
        SoSoValueAdapter(),
        CoinGlassAdapter(),
        CryptoQuantAdapter(),
        DeribitAdapter(),
        CMEAdapter(),
        ArkhamAdapter(),
        KaikoAdapter(),
        FearGreedAdapter(),
    ]


__all__ = ["all_adapters", "SourceAdapter"]
