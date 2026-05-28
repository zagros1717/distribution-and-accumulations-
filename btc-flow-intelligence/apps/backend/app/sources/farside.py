"""
Farside — US spot Bitcoin ETF daily net flows.

Farside publishes a public HTML table (farside.co.uk/btc). There is no official
JSON API, so live mode parses the page. Mock mode returns a realistic flow set.
"""

from __future__ import annotations

import re

from app.schemas import SignalReading
from app.sources.base import SourceAdapter


class FarsideAdapter(SourceAdapter):
    name = "farside"
    category = "etf_flows"
    required_keys = ()  # public page; live attempt only when mock_mode off

    @property
    def can_go_live(self) -> bool:
        from app.config import get_settings

        return not get_settings().mock_mode

    async def _fetch_live(self) -> list[SignalReading]:
        resp = await self._request(
            "GET", "https://farside.co.uk/btc/",
            headers={"User-Agent": "Mozilla/5.0 (BTC-Flow-Intelligence)"},
        )
        html = resp.text
        # Grab the "Total" net-flow figure from the most recent row.
        # Farside renders flows in $m; negative values are parenthesised.
        m = re.findall(r"\(?-?[\d,]+\.\d\)?", html)
        if not m:
            raise ValueError("could not parse Farside table")
        # Heuristic: the last large value is the latest total net flow.
        def parse(v: str) -> float:
            neg = v.startswith("(")
            num = float(v.strip("()").replace(",", ""))
            return -num if neg else num

        total = parse(m[-1])
        return [
            SignalReading(
                category="etf_flows", source=self.name, metric="etf_net_flow_usd_m",
                value=total, change_24h=None,
                score=self._score_from_change(total, 50, 250), raw={"parsed_from": "html"},
            )
        ]

    def _mock(self) -> list[SignalReading]:
        net_flow = self._r(-450, 650, 0)  # $m total daily net flow
        ibit = self._r(-120, 380, 0)
        fbtc = self._r(-80, 220, 0)
        gbtc = self._r(-160, 40, 0)       # GBTC tends to bleed
        return [
            SignalReading(category="etf_flows", source=self.name, metric="etf_net_flow_usd_m",
                          value=net_flow, change_24h=None,
                          score=self._score_from_change(net_flow, 50, 250),
                          raw={"IBIT": ibit, "FBTC": fbtc, "GBTC": gbtc, "unit": "$m"}),
        ]
