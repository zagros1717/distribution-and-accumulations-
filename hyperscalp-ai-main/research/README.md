# Research stack

Everything that should exist *before* a single dollar is risked.

```
research/
  data_recorder/   # Record HL L2 + tape + funding to parquet
  backtester/      # Replay parquets through the same alpha logic the executor uses
  ml/              # LightGBM classifier training + isotonic calibration
  optimization/    # Optuna hyper-search over backtester
  eval/            # Sharpe / Sortino / Calmar / DD / fees-as-%-of-PnL
```

Quick start:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r research/requirements.txt

# 1) Record some history (run for a day; longer is better)
python research/data_recorder/record_l2.py --coins BTC,ETH,SOL --out data/raw

# 2) Backtest each strategy on what you collected
python research/backtester/run.py --strategy range_mr     --data data/raw --out reports/range_mr
python research/backtester/run.py --strategy trend_breakout --data data/raw --out reports/trend_breakout
python research/backtester/run.py --strategy liquidation_fade --data data/raw --out reports/liquidation_fade

# 3) Walk-forward (60% train / 40% test, rolling)
python research/backtester/walk_forward.py --strategy trend_breakout --data data/raw

# 4) Train the ML gate (after enough trades are produced by backtests)
python research/ml/train_classifier.py --trades reports/all_trades.parquet --out models/profitability_v1.json

# 5) Hyperparam search
python research/optimization/optuna_search.py --strategy range_mr --trials 200 --data data/raw
```

What's production-ready vs. starting template:

- `data_recorder/record_l2.py` — production-ready, runs as a daemon
- `backtester/` — runs end-to-end on parquets; cost model is real (HL fees + slippage); the strategy ports are 1:1 with `base44/functions/botExecutor/strategies/`
- `ml/train_classifier.py` — runs end-to-end with LightGBM + isotonic calibration; needs trade history to train
- `optimization/optuna_search.py` — works but tune the search space for your time budget
- `eval/metrics.py` — final report metrics

Treat results from <1 month of recorded data with extreme skepticism. Walk-forward >3 months is when you can start trusting the numbers.
