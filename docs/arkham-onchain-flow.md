# Arkham BTC Exchange-Flow Monitor

This project now has a read-only Arkham on-chain monitor for daily BTC large-wallet flows into and out of exchanges.

It is separate from the L3/L2 order-book research pipeline. It does not trade, does not call exchange APIs, and does not move funds.

## What it measures

The monitor pulls BTC transfers from Arkham and classifies large transfers as:

- `exchange_inflow`: large wallet -> exchange
- `exchange_outflow`: exchange -> large wallet
- `ignored`: exchange-to-exchange, below-threshold, non-BTC, or transfers without an exchange side

The daily net flow is:

```text
net_inflow_btc = exchange_inflow_btc - exchange_outflow_btc
```

Interpretation:

- Positive net inflow: distribution pressure
- Negative net inflow: accumulation pressure
- Near zero: neutral / balanced

## Configuration

Settings live in `config/config.yaml` under `onchain.arkham`.

Important knobs:

```yaml
onchain:
  arkham:
    api_key_env: ARKHAM_API_KEY
    base_url: "https://api.arkm.com"
    transfers_path: "/transfers"
    min_transfer_btc: 100.0
    strong_net_btc: 500.0
    strong_net_share: 0.25
```

If Arkham changes parameter names or your account exposes a slightly different transfer query format, adjust these without changing code:

```yaml
start_time_param: startTime
end_time_param: endTime
limit_param: limit
pagination_cursor_param: cursor
extra_query_params: {}
```

## Local run

Set the Arkham API key as an environment variable:

```bash
export ARKHAM_API_KEY="your_key_here"
```

Run the last 24 hours:

```bash
python main.py onchain-flow
```

Run a fixed UTC window:

```bash
python main.py onchain-flow --start 2026-06-10T00:00:00Z --end 2026-06-11T00:00:00Z
```

Dry-run without calling Arkham:

```bash
python main.py onchain-flow --dry-run
```

Reports are written to:

```text
data/reports/onchain/
```

## Daily GitHub Actions run

The workflow `.github/workflows/arkham-onchain-flow.yml` runs every day at `00:15 UTC` and can also be started manually from the Actions tab.

Create this repository secret first:

```text
ARKHAM_API_KEY
```

The workflow then runs:

```bash
python main.py onchain-flow
```

and commits the generated Markdown report under `data/reports/onchain/`.

## Verdict logic

The monitor returns one of these values:

- `STRONG_DISTRIBUTION`
- `MILD_DISTRIBUTION`
- `STRONG_ACCUMULATION`
- `MILD_ACCUMULATION`
- `NEUTRAL_BALANCED`
- `NEUTRAL_NO_SIGNAL`

Strong verdicts require both:

1. `abs(net_inflow_btc) >= strong_net_btc`
2. `abs(net_inflow_btc) / (inflow_btc + outflow_btc) >= strong_net_share`

With the default config, that means a net flow of at least `500 BTC` and at least `25%` dominance over total classified flow.

## Safety boundary

This is a read-only analytics integration. It intentionally does not add exchange account access, order placement, withdrawals, or trading execution.
