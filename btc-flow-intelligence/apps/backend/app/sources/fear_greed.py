"""
Fear & Greed Index — alternative.me publishes a free, keyless JSON endpoint,
so this adapter runs live whenever mock_mode is off.
"""

from __future__ import annotations

from app.schemas import SignalReading
from app.sources.base import SourceAdapter


class FearGreedAdapter(SourceAdapter):
    name = "fear_greed"
    category = "sentiment"
    required_keys = ()

    @property
    def can_go_live(self) -> bool:
        from app.config import get_settings

        return not get_settings().mock_mode

    async def _fetch_live(self) -> list[SignalReading]:
        resp = await self._request("GET", "https://api.alternative.me/fng/?limit=2")
        data = resp.json()["data"]
        today = int(data[0]["value"])
        yday = int(data[1]["value"]) if len(data) > 1 else today
        return [
            SignalReading(
                category="sentiment", source=self.name, metric="fear_greed_index",
                value=today, change_24h=today - yday,
                # contrarian-lite: extreme fear = accumulation opportunity,
                # extreme greed = distribution risk. We score the *deviation* from 50.
                score=self._score_from_change(50 - today, 12, 28),
                raw={"classification": data[0]["value_classification"]},
            )
        ]

    def _mock(self) -> list[SignalReading]:
        idx = self._r(20, 82, 0)
        chg = self._r(-12, 12, 0)
        cls = (
            "Extreme Fear" if idx < 25 else
            "Fear" if idx < 45 else
            "Neutral" if idx < 55 else
            "Greed" if idx < 75 else "Extreme Greed"
        )
        return [
            SignalReading(category="sentiment", source=self.name, metric="fear_greed_index",
                          value=idx, change_24h=chg,
                          score=self._score_from_change(50 - idx, 12, 28),
                          raw={"classification": cls}),
        ]
