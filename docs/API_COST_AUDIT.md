# API cost audit: CryptoQuant / CoinGlass replacement plan

## Current project posture

The existing project is already designed as an offline research machine. The core recorder uses public Bitfinex and Coinbase market-data endpoints, stores raw frames before normalization, and keeps execution permanently disabled. I did not find a direct CryptoQuant or CoinGlass dependency in the repository search index; the expensive-data problem is therefore best solved as an optional market-context sidecar rather than by changing the L3 recorder.

## What was implemented

A new `src/market_data` package and `scripts/collect_market_data.py` command collect lower-cost public market/context data and write it under:

```text
data/market_metrics/source=<provider>/metric=<metric>/symbol=<symbol>/date=YYYY-MM-DD/metrics.jsonl
```

The collector uses an on-disk HTTP cache at `data/http_cache` so dashboards, notebooks, or cron jobs do not repeatedly hit the same endpoint. TTLs are configured per metric in `config/config.yaml`.

### Provider coverage

| Expensive vendor use case | Replacement implemented | Notes |
|---|---|---|
| CoinGlass open interest | Binance Futures + Bybit public APIs | Venue-specific, auditable, no aggregation black box. |
| CoinGlass funding rates | Binance Futures + Bybit public APIs | Keep both venues to avoid single-venue bias. |
| CoinGlass long/short ratio | Binance global long/short + Bybit account ratio | Not identical to CoinGlass aggregate, but usually good enough for research features. |
| General price / market cap / volume | CoinGecko market chart | Cheap context data; not L3 data. |
| DeFi / stablecoin context | DefiLlama public endpoints | Low-frequency context, cached hourly. |
| CryptoQuant entity-labelled exchange flows, miner flows, whale labels | Not fully replaceable for free | Keep paid calls only for these unique labelled metrics, and cache snapshots. |

## Operational recommendation

1. Run the L3 recorder as before for raw order-book research.
2. Run `python scripts/collect_market_data.py --config config/config.yaml` every 1-5 minutes for derivatives context.
3. Run the same command hourly or daily for DefiLlama context.
4. Keep CryptoQuant disabled by default. If you later add it, route it through the same cache layer and store paid snapshots, never live dashboard calls.

## Risk notes

- Exchange public endpoints can change rate limits or temporarily restrict regions. The collector logs per-provider failures and continues unless `market_data.fail_fast` is set to true.
- Exchange-derived long/short metrics are not the same as a multi-exchange CoinGlass aggregate. Treat them as venue-specific features.
- CryptoQuant's most valuable product is labelled entity intelligence. Public APIs cannot perfectly reproduce exchange reserves, miner flows, whale labels, or SOPR-style labelled metrics without maintaining your own address-label pipeline.

## Cost-control checklist

- Cache all HTTP responses with metric-specific TTLs.
- Store normalized metrics locally and read from disk for dashboards.
- Add paid vendors only behind a provider interface, disabled by default.
- Do not expose paid API calls to frontend refreshes.
- Keep private exchange API access disabled; this is research-only infrastructure.
