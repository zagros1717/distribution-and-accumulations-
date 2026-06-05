from __future__ import annotations

from datetime import datetime, timezone

from src.context24.calibration import Calibration
from src.context24.scoring import score_table


def _row(source, metric, value, delta=None, unit="", as_of="2026-01-01T00:00:00Z"):
    return {
        "source": source,
        "metric": metric,
        "value": value,
        "delta_24h": delta,
        "unit": unit,
        "as_of": as_of,
        "fetched_at": "2026-01-01T00:01:00Z",
    }


def test_rejects_rows_without_calibration_even_if_values_look_strong():
    signal = score_table([
        _row("fear_greed", "fear greed index", 12, unit="index"),
        _row("coinglass", "funding rate", 0.03, unit="pct"),
    ], calibration={}, now=datetime(2026, 1, 1, 0, 2, tzinfo=timezone.utc))

    assert signal.status == "REJECTED"
    assert "no_usable_calibrated_rows" in signal.reasons
    assert all(not r.usable for r in signal.rows)


def test_missing_required_delta_forces_unusable():
    calibration = {
        "coinglass.open_interest": Calibration("coinglass.open_interest", 100, 0.6, 0.6, 0.01, -0.01, 0.8)
    }
    signal = score_table([
        _row("coinglass", "open interest", 21.29, delta=None, unit="usd_b"),
    ], calibration=calibration, now=datetime(2026, 1, 1, 0, 2, tzinfo=timezone.utc))

    row = signal.rows[0]
    assert row.usable is False
    assert "missing_required_delta_24h" in row.reasons


def test_confirmed_requires_three_independent_categories():
    calibration = {
        "coinbase.orderbook_imbalance": Calibration("coinbase.orderbook_imbalance", 100, 0.6, 0.6, 0.01, -0.01, 0.9),
        "farside.etf_net_flow_usd_m": Calibration("farside.etf_net_flow_usd_m", 100, 0.6, 0.6, 0.01, -0.01, 0.9),
        "coinglass.liquidation_skew": Calibration("coinglass.liquidation_skew", 100, 0.6, 0.6, 0.01, -0.01, 0.9),
    }
    signal = score_table([
        _row("coinbase", "orderbook imbalance", 2.0, unit="z"),
        _row("farside", "etf net flow usd m", 500.0, unit="usd_m"),
        _row("coinglass", "liquidation skew", -5.0, unit="pct"),
    ], calibration=calibration, now=datetime(2026, 1, 1, 0, 2, tzinfo=timezone.utc))

    assert signal.status == "CONFIRMED_LONG"
    assert signal.final_score > 0
    assert len(signal.category_scores) == 3
