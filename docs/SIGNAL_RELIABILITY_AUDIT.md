# Signal reliability audit

## Summary

The project already had several good safeguards: public-only data capture, raw-frame preservation, deterministic replay ordering, invalid-book filtering, cost-aware labels, walk-forward validation, and OOS-only backtests. This audit tightened the places where a weak signal could still look reliable.

## Main issues found and fixed

### 1. Variable-horizon labels during data gaps

Old behavior: label generation used the first future row after `T + horizon` with a tolerance as large as the full horizon. During a gap, a 5-second label could effectively become close to a 10-second label.

Fix: labels now require the future row to be within one feature interval after the target timestamp. Rows with missing or late future prices are dropped, and the label output records `future_ts`, `actual_horizon_s`, and `horizon_error_ms` for auditability.

### 2. Raw price-level features could make signals nonstationary

Old behavior: absolute levels such as `mid_price`, `best_bid`, `best_ask`, `last_trade_price`, `vwap`, and `microprice` were eligible model features. Those can make XGBoost learn BTC price regime rather than order-book behavior.

Fix: trainer excludes absolute price-level columns from model inputs while keeping relative/microstructure features such as spread, relative spread, order-book imbalance, depth, flow counts, returns, and volatility.

### 3. Weak class coverage could still produce a model artifact

Old behavior: if there were very few long or short labels, training could still proceed and produce a model that looked operational but had no statistical basis for directional signals.

Fix: training now rejects the full dataset if either long or short labels are below a minimum threshold, skips walk-forward folds with inadequate directional coverage, and records `rejection_reasons` in metadata.

### 4. Backtests could execute across stale snapshots

Old behavior: entry and exit used the next snapshot after the target time, even if it arrived much later because of a gap.

Fix: backtest now infers a snapshot freshness tolerance from cadence, or uses `BacktestConfig.max_snapshot_delay_ms` when set. Candidate trades are skipped if entry or exit snapshots are stale or invalid. The summary reports stale-entry, stale-exit, and invalid-execution skips.

### 5. Rejections were not visible enough

Old behavior: reports rejected based on PnL/drawdown/data quality, but trainer-level reliability rejections were not first-class report reasons.

Fix: daily reports now include trainer reliability rejections and stale-execution skips in the decision section.

## Signal acceptance checklist

A signal should be considered only if all of these are true:

- Data quality is acceptable: low missing periods and low corrupted-book percentage.
- Labels have strict horizon matching, with low `horizon_error_ms`.
- There are enough long and short examples in the full dataset and in walk-forward folds.
- Walk-forward validation has more than one fold.
- OOS predictions exist only for validation windows.
- Backtest trades are not blocked primarily by stale execution snapshots.
- Net PnL remains positive after fees, spread, slippage, cooldown, latency, and trade caps.
- Feature importance is not dominated by suspicious time/index/proxy columns.

## Remaining limitations

- This is still research, not a production trading system.
- Backtest fills remain simplified: they use top-of-book mid/spread plus slippage, not full queue-position simulation.
- Public venue data can drop or reorder messages; the replay pipeline marks invalid windows, but it cannot recover information never received.
- The model is only as reliable as the amount and diversity of recorded data. A few days of BTC data is not enough to trust a live strategy.

## Operational recommendation

After this change, regenerate labels and models rather than reusing old label partitions:

```bash
python main.py replay --exchange bitfinex --symbol BTCUSD --start YYYY-MM-DD --end YYYY-MM-DD
python main.py features --exchange bitfinex --symbol BTCUSD --start YYYY-MM-DD --end YYYY-MM-DD
python main.py labels --exchange bitfinex --symbol BTCUSD --start YYYY-MM-DD --end YYYY-MM-DD
python main.py train --exchange bitfinex --symbol BTCUSD --horizon 5 --start YYYY-MM-DD --end YYYY-MM-DD
python main.py backtest --exchange bitfinex --symbol BTCUSD --horizon 5 --start YYYY-MM-DD --end YYYY-MM-DD
python main.py report --exchange bitfinex --symbol BTCUSD --horizon 5 --start YYYY-MM-DD --end YYYY-MM-DD
```

If `train` exits nonzero or report says `REJECTED`, treat the signal as unreliable.
