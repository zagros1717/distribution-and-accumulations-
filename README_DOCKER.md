# Running BTC Research Machine with Docker

This project is research-only. It does not place trades.

## Build

```bash
docker compose build
```

## Show CLI help

```bash
docker compose run --rm btc-research
```

## Run tests

```bash
docker compose run --rm btc-research pytest -q
```

## Start live recorder

This connects to Bitfinex first and falls back to Coinbase according to `config/config.yaml`.
Data is written to `./data` on the host.

```bash
docker compose up recorder
```

To stop:

```bash
docker compose down
```

## Run offline pipeline

After collecting data for a date, replace the date/symbol as needed:

```bash
docker compose run --rm btc-research \
  python main.py pipeline \
  --exchange bitfinex \
  --symbol BTCUSD \
  --start 2026-05-23 \
  --end 2026-05-23 \
  --horizon 5
```

For Coinbase fallback data:

```bash
docker compose run --rm btc-research \
  python main.py pipeline \
  --exchange coinbase \
  --symbol BTC-USD \
  --start 2026-05-23 \
  --end 2026-05-23 \
  --horizon 5
```
