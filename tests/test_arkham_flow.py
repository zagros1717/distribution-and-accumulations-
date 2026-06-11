from datetime import datetime, timezone

from src.onchain.arkham_flow import analyze_transfers, classify_transfer, config_from_dict


def _cfg(**overrides):
    base = {
        "onchain": {
            "arkham": {
                "min_transfer_btc": 100.0,
                "strong_net_btc": 500.0,
                "strong_net_share": 0.25,
            }
        }
    }
    base["onchain"]["arkham"].update(overrides)
    return config_from_dict(base)


def test_classifies_large_wallet_to_exchange_as_inflow():
    cfg = _cfg()
    transfer = {
        "timestamp": "2026-06-11T00:00:00Z",
        "txHash": "abc",
        "token": {"symbol": "BTC"},
        "unitValue": 250.0,
        "from": {"address": "bc1whale", "name": "Unknown Whale"},
        "to": {
            "address": "bc1binance",
            "name": "Binance Deposit Wallet",
            "entity": {"name": "Binance", "type": "exchange"},
        },
    }

    out = classify_transfer(transfer, cfg)

    assert out.direction == "exchange_inflow"
    assert out.reason == "large_wallet_to_exchange"
    assert out.amount_btc == 250.0


def test_classifies_exchange_to_large_wallet_as_outflow():
    cfg = _cfg()
    transfer = {
        "timestamp": "2026-06-11T00:05:00Z",
        "txHash": "def",
        "token": {"symbol": "BTC"},
        "unitValue": 300.0,
        "from": {
            "address": "bc1coinbase",
            "name": "Coinbase Prime",
            "entity": {"name": "Coinbase", "type": "exchange"},
        },
        "to": {"address": "bc1coldwallet", "name": "Unknown Cold Wallet"},
    }

    out = classify_transfer(transfer, cfg)

    assert out.direction == "exchange_outflow"
    assert out.reason == "exchange_to_large_wallet"
    assert out.amount_btc == 300.0


def test_ignores_exchange_to_exchange_transfers():
    cfg = _cfg()
    transfer = {
        "token": {"symbol": "BTC"},
        "unitValue": 500.0,
        "from": {"name": "Binance Hot Wallet", "entity": {"type": "exchange"}},
        "to": {"name": "Coinbase Prime", "entity": {"type": "exchange"}},
    }

    out = classify_transfer(transfer, cfg)

    assert out.direction == "ignored"
    assert out.reason == "exchange_to_exchange"


def test_ignores_sub_threshold_transfers():
    cfg = _cfg()
    transfer = {
        "token": {"symbol": "BTC"},
        "unitValue": 99.99,
        "from": {"name": "Unknown Whale"},
        "to": {"name": "Binance"},
    }

    out = classify_transfer(transfer, cfg)

    assert out.direction == "ignored"
    assert out.reason == "below_min_transfer_btc"


def test_daily_summary_distribution_verdict():
    cfg = _cfg(strong_net_btc=500.0, strong_net_share=0.25)
    transfers = [
        {
            "token": {"symbol": "BTC"},
            "unitValue": 1000.0,
            "from": {"name": "Unknown Whale"},
            "to": {"name": "Binance", "entity": {"type": "exchange"}},
        },
        {
            "token": {"symbol": "BTC"},
            "unitValue": 200.0,
            "from": {"name": "Coinbase", "entity": {"type": "exchange"}},
            "to": {"name": "Unknown Cold Wallet"},
        },
    ]

    summary = analyze_transfers(
        transfers,
        datetime(2026, 6, 10, tzinfo=timezone.utc),
        datetime(2026, 6, 11, tzinfo=timezone.utc),
        cfg,
    )

    assert summary.inflow_btc == 1000.0
    assert summary.outflow_btc == 200.0
    assert summary.net_inflow_btc == 800.0
    assert summary.verdict == "STRONG_DISTRIBUTION"
