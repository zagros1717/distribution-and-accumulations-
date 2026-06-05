# 24h market context scoring engine

This engine replaces the dashboard label `Confirmed 24h Signals` with a stricter point-in-time scoring process.

## What it does

- Parses dashboard rows from CSV.
- Checks every row against a metric registry.
- Rejects missing values, stale rows, missing timestamps, unit mismatches, and required missing deltas.
- Requires historical calibration before any row can influence the final score.
- Caps category influence so one vendor/category cannot dominate.
- Requires at least three independent calibrated categories before a signal can be confirmed.
- Writes a Markdown audit report with row-level reasons.

## Input CSV

Required columns:

```text
source,metric,value,delta_24h,unit,as_of,fetched_at
```

Example:

```csv
source,metric,value,delta_24h,unit,as_of,fetched_at
coinbase,orderbook imbalance,2.1,,z,2026-01-01T00:00:00Z,2026-01-01T00:01:00Z
farside,etf net flow usd m,500,,usd_m,2026-01-01T00:00:00Z,2026-01-01T00:01:00Z
coinglass,liquidation skew,-5,,pct,2026-01-01T00:00:00Z,2026-01-01T00:01:00Z
```

## Historical calibration CSV

Required columns:

```text
metric_key,value,forward_return_24h
```

Example:

```csv
metric_key,value,forward_return_24h
coinbase.orderbook_imbalance,1.5,0.012
coinbase.orderbook_imbalance,-1.2,-0.006
farside.etf_net_flow_usd_m,500,0.009
```

Without calibration, rows are treated as unproven and cannot create a confirmed signal.

## Run

```bash
python scripts/score_context24.py \
  --input data/context24/current.csv \
  --history data/context24/history.csv \
  --out data/reports/context24.md
```

The command exits with:

- `0` only for `CONFIRMED_LONG` or `CONFIRMED_SHORT`
- `1` for `REJECTED` or `WATCH`

## Status meanings

- `REJECTED`: not enough usable calibrated evidence, stale/missing data, or too few independent categories.
- `WATCH`: some usable evidence exists, but confidence/category agreement is not enough.
- `CONFIRMED_LONG`: enough calibrated categories agree bullishly.
- `CONFIRMED_SHORT`: enough calibrated categories agree bearishly.

## Why your previous table was not confirmed

The old table allowed strong symbols like `++` or `--` even when `delta_24h` was missing, timestamps were unknown, or no historical edge was attached. This engine forces those rows to be unusable or low confidence until they are fresh, correctly typed, and historically calibrated.
