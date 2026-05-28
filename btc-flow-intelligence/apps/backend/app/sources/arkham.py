"""
Arkham — entity-level flows: whale transfers, ETF custody wallets, government
holdings, and named treasury wallets (Strategy, Tesla, SpaceX).

Live mode requires an Arkham API key. Mock mode synthesises a plausible
entity-flow picture.
"""

from __future__ import annotations

from app.config import get_settings
from app.schemas import SignalReading
from app.sources.base import SourceAdapter

settings = get_settings()


class ArkhamAdapter(SourceAdapter):
    name = "arkham"
    category = "entity_flows"
    required_keys = ("arkham_api_key",)
    BASE = "https://api.arkhamintelligence.com"

    async def _fetch_live(self) -> list[SignalReading]:
        headers = {"API-Key": settings.arkham_api_key or ""}
        resp = await self._request(
            "GET",
            f"{self.BASE}/transfers?base=bitcoin&flow=all&timeLast=86400&limit=100",
            headers=headers,
        )
        transfers = resp.json().get("transfers", [])
        to_exch = sum(t.get("unitValue", 0) for t in transfers if t.get("toIsExchange"))
        from_exch = sum(t.get("unitValue", 0) for t in transfers if t.get("fromIsExchange"))
        net = from_exch - to_exch  # positive = net withdrawal from exchanges (accumulation)
        return [
            SignalReading(
                category="entity_flows", source=self.name, metric="whale_net_exchange_flow",
                value=round(net, 0), change_24h=None,
                score=self._score_from_change(net / 100, 5, 15),
                raw={"to_exchange": to_exch, "from_exchange": from_exch},
            )
        ]

    def _mock(self) -> list[SignalReading]:
        whale_net = self._r(-6000, 6000, 0)        # BTC net off-exchange by whales
        etf_custody_chg = self._r(-2500, 4500, 0)  # BTC into ETF custody (Coinbase Prime)
        gov_change = self._r(-1500, 200, 0)        # govt wallets (US/DE) tend to distribute
        strategy_change = self._r(0, 9000, 0)      # Strategy (MSTR) only buys
        tesla_change = 0                            # Tesla static recently
        spacex_change = 0
        return [
            SignalReading(category="entity_flows", source=self.name, metric="whale_net_exchange_flow",
                          value=whale_net, change_24h=None,
                          score=self._score_from_change(whale_net / 1000, 1, 3),
                          raw={"unit": "BTC", "note": "positive = leaving exchanges"}),
            SignalReading(category="entity_flows", source=self.name, metric="etf_custody_net_change",
                          value=etf_custody_chg, change_24h=None,
                          score=self._score_from_change(etf_custody_chg / 1000, 0.8, 2.5), raw={}),
            SignalReading(category="entity_flows", source=self.name, metric="government_wallet_change",
                          value=gov_change, change_24h=None,
                          score=self._score_from_change(-gov_change / 200, 2, 5), raw={}),
            SignalReading(category="entity_flows", source=self.name, metric="treasury_wallet_change",
                          value=strategy_change, change_24h=None,
                          score=self._score_from_change(strategy_change / 2000, 0.5, 2.0),
                          raw={"Strategy": strategy_change, "Tesla": tesla_change, "SpaceX": spacex_change}),
        ]
