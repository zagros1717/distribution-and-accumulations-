# btc_research_machine

An **offline** Bitcoin order-book research and training machine. Collects raw market data, reconstructs the L3 order book, builds cost-aware features and labels, trains XGBoost on walk-forward folds, runs a realistic backtest, and writes a daily report.

It does **not** trade. It cannot trade. Execution is permanently disabled by design — see *Safety* below.

---

## What it does, in one sentence

Answers the question *"can recent L3/L2 order-book behavior predict whether BTC's mid-price will move enough over the next short horizon to overcome fees, spread, and realistic execution costs?"* — and tells you honestly whether the answer is yes or no for your data.

---

## Requirements

- Python 3.11+
- ~5 GB free disk per day of live recording (raw + normalized parquet)
- Internet egress to `api-pub.bitfinex.com` (primary) and `ws-feed.exchange.coinbase.com` + `api.exchange.coinbase.com` (fallback)

Install:

```bash
pip install -r requirements.txt
```

---

## Project layout

```
btc_research_machine/
├── config/config.yaml          # all knobs live here
├── data/                       # created on first run
│   ├── raw/                    # exact websocket frames pre-normalization,
│   │                           #   partitioned by exchange/symbol/UTC date
│   ├── normalized/             # canonical BookEvent rows
│   ├── snapshots/              # book snapshots, partitioned by date and interval_ms
│   ├── features/               # feature parquet, partitioned by interval_ms first
│   ├── labels/                 # labels — partitioned by interval_ms AND horizon_s,
│   │                           #   because 100ms-features-with-5s-horizon labels are
│   │                           #   NOT the same dataset as 1000ms-features-with-5s-horizon
│   ├── models/                 # trained XGBoost models + per-fold OOS predictions
│   ├── metadata/               # source-switch markers (which exchange was live when)
│   └── reports/                # daily Markdown reports
├── src/
│   ├── schema.py               # canonical BookEvent dataclass + arrow schemas
│   ├── safety.py               # the "no trading" enforcement
│   ├── recorder.py             # live Bitfinex→Coinbase failover recorder
│   │                           #   (Event-based watchdog; raw-frame capture)
│   ├── adapters/               # BitfinexRawBookAdapter, CoinbaseFullBookAdapter
│   ├── book/                   # L3 order-book engine + replayer
│   │                           #   (match_fill, reset, corruption resync)
│   ├── features/               # interval-aware feature engine (no look-ahead)
│   ├── labels/                 # round-trip-cost-aware ±1/0 labels
│   ├── models/                 # walk-forward XGBoost trainer + evaluator
│   │                           #   (no scaler — XGB is invariant to monotonic transforms)
│   ├── backtest/               # OOS-only simulator using fold validation predictions
│   ├── reports/                # Markdown daily report writer
│   ├── storage/                # parquet writers (DatePartitionedWriter) + DuckDB views
│   └── utils/                  # config, logging, time, validation
├── tests/                      # pytest — order book, normalization, anti-leakage,
│                               #   partial fills, watchdog failover, scaler removal,
│                               #   interval label isolation
├── main.py                     # CLI entry point
├── pytest.ini                  # asyncio mode for recorder tests
└── requirements.txt
```

---

## Running it

The CLI lives in `main.py`. All subcommands read `config/config.yaml` first, which is where the safety check runs.

### 1. Live record (Bitfinex with Coinbase fallback)

```bash
python main.py record
```

Connects to Bitfinex BTC/USD raw book + trades. If Bitfinex fails repeatedly (configurable in `recorder.max_reconnects_before_switch`), the recorder switches to Coinbase BTC-USD full channel and writes a source-switch marker to `data/metadata/source_switches/`.

Raw messages go to `data/raw/{exchange}/{symbol}/date=YYYY-MM-DD/`. Normalized `BookEvent` rows go to `data/normalized/exchange={ex}/symbol={sym}/date=YYYY-MM-DD/`.

Stop with `Ctrl-C` — the recorder flushes buffers on `SIGINT`/`SIGTERM`.

### 2. Reconstruct snapshots

```bash
python main.py replay --exchange bitfinex --symbol BTCUSD --start 2026-05-22 --end 2026-05-22
```

Reads normalized events, drives the L3 book, and writes 1-second snapshots (`best_bid`, `best_ask`, `mid`, `spread`, top-N depth, `is_valid` flag) to `data/snapshots/`.

### 3. Generate features

```bash
python main.py features --exchange bitfinex --symbol BTCUSD --start 2026-05-22 --end 2026-05-22
```

Produces L1, depth/imbalance, L3 order-flow counts, trade-flow, volatility, and microprice features bucketed at the interval in `config.features.intervals_ms` (default 1000 ms). Rolling features are computed via `shift(k)` then `rolling(k)` to make look-ahead structurally impossible. Window sizes are **interval-aware**: a "5-second return" at 1000ms cadence is `shift(5)`, and at 100ms cadence becomes `shift(50)`, so feature semantics stay constant as the snapshot rate changes.

### 4. Generate labels

```bash
python main.py labels --exchange bitfinex --symbol BTCUSD --start 2026-05-22 --end 2026-05-22
```

For each horizon in `config.labels.horizons_seconds` (default 1s, 5s, 10s, 30s), each row gets a cost-aware classification target. The threshold is the **round-trip** cost — `2 × (taker_fee + half_spread_buffer + slippage_buffer)` — because a trade has to overcome those costs on both entry AND exit to actually profit. With the default config that's `2 × (10 + 3 + 2) = 30 bps`. Then:

- `+1` if `future_return > +threshold`
- `-1` if `future_return < -threshold`
- ` 0` otherwise

Labels are written to `data/labels/interval_ms=<N>/horizon_s=<H>/...`. The interval partition matters: labels generated from 100ms-cadence features are *not* the same dataset as labels generated from 1000ms-cadence features, and mixing them at training time produces noise. This threshold is kept consistent with the backtester's cost model — see `compute_threshold_bps()` in `src/labels/label_engine.py`.

### 5. Train

```bash
python main.py train --exchange bitfinex --symbol BTCUSD --horizon 5
```

Walk-forward XGBoost: 10-day train → 1-day val, sliding by 1 day. **No feature scaling** is applied — XGBoost is invariant to monotonic feature transformations, and skipping the scaler eliminates a class of save/load bugs (where the model is persisted but the scaler isn't, silently corrupting inference). Per-fold metrics (accuracy, per-class precision/recall, log-loss, confusion, feature importance by gain) are saved alongside the final-fold model and a `oos_predictions.parquet` file in `data/models/xgboost/horizon_<H>s/`.

### 6. Backtest (out-of-sample only)

```bash
python main.py backtest --exchange bitfinex --symbol BTCUSD --horizon 5 --start 2026-05-22 --end 2026-05-22
```

Reads `oos_predictions.parquet` (the per-fold validation predictions saved by `train`) and inner-joins it onto the feature timeline. Because that file only contains rows from each fold's *validation* window — never its training window — the backtest is provably out-of-sample by construction. There's no inference happening here, and no way for in-sample predictions to leak in.

The simulator applies:

- entry latency (config: `backtest.latency_ms`, default 250 ms)
- taker fees on both legs
- half-spread + slippage estimate on entry and exit
- a minimum confidence filter (`backtest.min_confidence`)
- per-day trade cap and cooldown between signals

Outputs trade-level PnL, win rate, max drawdown, an approximate Sharpe, and a daily PnL series. If you run `backtest` without first running `train`, it refuses to proceed (no in-sample fallback path).

### 7. Daily report

```bash
python main.py report --exchange bitfinex --symbol BTCUSD --horizon 5 --start 2026-05-22 --end 2026-05-22
```

Produces a Markdown report in `data/reports/` covering data quality (raw message count, missing periods, corrupted-book %), model metrics (per-fold table, last-fold confusion, top features), backtest results, and an **ACCEPTED / REJECTED** decision with reasons. Auto-reject criteria from `config.report.reject_if`:

- net PnL after costs negative
- max drawdown above configured bps
- corrupted-book % above configured threshold
- result only works on one day (no stability across folds)

### 8. End-to-end pipeline

```bash
python main.py pipeline --exchange bitfinex --symbol BTCUSD --horizon 5 --start 2026-05-22 --end 2026-05-22
```

Runs replay → features → labels → train → backtest → report in sequence.

---

## Safety

This project has **zero trading capability**. There is no path to a trading call anywhere in the codebase. The recorder uses *only* public WebSocket endpoints. There are no private API keys, no order placement, no account access, no withdrawals.

`src/safety.py` is called inside `load_config()`, so every entry point — including the CLI, the recorder, and the test fixtures — passes through it. If anyone flips `safety.execution_enabled`, `safety.allow_private_api`, or `safety.allow_withdrawals` to `true` in `config.yaml`, the code refuses to load with `ExecutionDisabledError`.

These flags are belt-and-braces. The real protection is that the trading code does not exist.

---

## Anti-leakage

Look-ahead bias is the failure mode that kills HFT research projects. The code guards against it in several places:

1. **Feature bucketing**: every event is assigned to the snapshot bucket it *completes*, via `dt.ceil(interval)`. An event at `T = 1.001s` lands in the `[1.001s, 2.000s]` bucket — not the `[0s, 1s]` one.
2. **Rolling features**: computed as `series.shift(k).rolling(k)` so row *i* never sees row *i* itself, let alone row *i+1*. Window sizes are interval-aware (5 seconds at 100ms = 50 rows, at 1000ms = 5 rows).
3. **Walk-forward splits**: `assert_train_before_val` runs on every fold; no scaler exists to be misused.
4. **Out-of-sample backtest**: the simulator reads `oos_predictions.parquet` which only contains rows from each fold's validation window. In-sample contamination is impossible because in-sample rows aren't in the file.
5. **Round-trip label thresholds**: labels are tagged `±1` only if the future return clears the full round-trip cost — fees + half-spread + slippage on entry *and* exit — so a positive label means "a trade could have profited," not "the price moved a bit." This keeps labels consistent with what the backtester actually pays.

The `tests/test_no_lookahead.py` suite verifies the **truncated-frame property** directly: for each rolling kernel used in the feature engine, the value at row *i* on the full frame must equal the value at row *i* on the frame truncated to rows ≤ *i*. There's also a negative-control test that fails on a deliberately-buggy kernel, so the test has teeth.

Run the suite:

```bash
pytest tests/ -v
```

---

## Bitfinex and Coinbase are not the same market

Bitfinex BTC/USD and Coinbase BTC-USD trade on different venues with different liquidity, different fees, different tick sizes, and different microstructure. The recorder treats them as fallback sources for *availability*, not as substitutes for *data quality*.

Their data is partitioned separately under `data/raw/bitfinex/BTCUSD/` and `data/raw/coinbase/BTC-USD/`. Train **separate models** per exchange — or, if you want a single model, include `exchange` as an explicit feature and accept that the model is learning a venue effect alongside microstructure. Merging the two as if they were one stream will silently corrupt your results.

---

## Mode B: historical backfill (not built in v1)

For dates before you started recording, the codebase is structured so you can drop in a vendor adapter (Tardis.dev, Kaiko, CoinAPI, Crypto Lake, etc.) that emits `BookEvent` rows in the same canonical schema and writes them via `NormalizedEventStore`. Once the rows are in `data/normalized/`, every downstream stage (`replay`, `features`, `labels`, `train`) works identically. This is intentionally an extension point — v1 records its own dataset forward in time.

---

## MVP success conditions

The first version isn't trying to make money. It's trying to be *honest*. v1 is a success when:

1. The recorder collects data for 24 hours uninterrupted.
2. The reconstructor replays the day without corruption (or only brief, flagged corrupt periods).
3. Features are generated for the full day with no look-ahead bias (tests pass).
4. XGBoost trains walk-forward on the data.
5. The daily report renders a verdict — **ACCEPTED** or **REJECTED** — and the reasoning is auditable.

If the verdict on a 5-second horizon is REJECTED with "PnL negative after costs", that's not a failure of the system. That's the system doing its job.

---

## License & disclaimer

Research code, not financial advice. Not a trading system. Read the safety section twice if you're tempted to extend this.
