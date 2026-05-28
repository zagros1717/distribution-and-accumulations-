# BTC Research Machine Dashboard

This project now includes a lightweight Streamlit dashboard.

## Run locally with Docker

```bash
docker compose build
docker compose up dashboard
```

Open:

```text
http://localhost:8501
```

## Recommended workflow

In one terminal, run the recorder:

```bash
docker compose up -d recorder
```

In another terminal, run the dashboard:

```bash
docker compose up dashboard
```

After you have collected enough data, run the pipeline for the date you want:

```bash
docker compose run --rm btc-research \
  python main.py pipeline \
  --exchange bitfinex \
  --symbol BTCUSD \
  --start 2026-05-23 \
  --end 2026-05-23 \
  --horizon 5
```

Then refresh the dashboard.

## What the dashboard shows

- Raw messages and normalized events
- Recorder log tail
- Event type distribution
- Order-book snapshots: mid price, spread, depth
- Feature previews and feature charts
- Label distribution
- XGBoost metadata and fold metrics
- OOS prediction probabilities
- Markdown reports
- Data folder sizes

This dashboard is read-only. It does not place trades and does not access private exchange APIs.
