# Hyperscalp v6 — multi-strategy rebuild

This document describes what changed from v5, why, and how to roll the new system out safely.

## Summary of changes

The v5 executor was a single composite-score engine with three "modes" (relaxed/moderate/strict) that were really just three confidence thresholds over the same alphas. The new system replaces it with three orthogonal strategies, each with its own thesis, regime, and exit logic, plus the missing infrastructure (cost model, vol-targeted sizing, backtester, ML gate, walk-forward, allocator).

```
v5 = single engine, 3 thresholds, 1 timeframe, no costs, no backtest
v6 = 3 strategies × HTF context × cost-aware × vol-targeted × bandit allocated × backtested × calibrated
```

## File layout

```
base44/
  entities/
    BotConfig.jsonc         REWRITTEN — multi-strategy config
    StrategyState.jsonc     NEW — rolling per-strategy stats for the bandit
    Trade.jsonc             EXTENDED — strategy field, partial-TP, trail, OIDs
    Liquidation.jsonc       NEW — forced-liquidation events (fed by tapeCollector)
  functions/
    botExecutor/
      entry.ts              REWRITTEN — multi-strategy dispatcher
      lib/
        hl.ts               Hyperliquid client (signing, IOC, post-only, triggers, data)
        features.ts         HTF, ADX, Choppiness, VWAP, CVD divergence, large prints, sweeps
        regime.ts           TREND_UP / TREND_DOWN / RANGE / HIGH_VOL / NO_TRADE
        sizing.ts           Vol-target + Kelly + portfolio budget
        costs.ts            Round-trip cost + edge filter (HL fees + slippage)
        execution.ts        IOC / post-only / depth-aware modes
        allocator.ts        Multi-armed bandit weights from rolling Sharpe
      strategies/
        types.ts            Common Signal/Context types
        rangeMR.ts          A — VWAP fade in chop
        trendBreakout.ts    B — HTF trend continuation w/ CVD confirm
        liquidationFade.ts  C — fade liquidation cascades

research/                    NEW — research stack (Python)
  data_recorder/record_l2.py   record HL trades/L2/funding/liquidations to parquet
  backtester/
    cost_model.py             mirror of TS costs
    data_loader.py            parquet → bars
    features.py               python ports of feature pipeline (keep in sync!)
    engine.py                 vectorized event-driven backtest
    run.py                    backtest one strategy
    walk_forward.py           rolling-window in-sample/out-of-sample
    strategies/               python ports of the three strategies
  eval/metrics.py             Sharpe / Sortino / Calmar / DD / fees-as-%-of-PnL
  ml/train_classifier.py      LightGBM + isotonic calibration
  optimization/optuna_search.py hyperparam search
  README.md                   stack-specific quickstart
```

## Why each piece exists (rationale traceable to the v5 critique)

| Critique of v5 | What v6 does about it |
|---|---|
| Three "modes" share alphas, not orthogonal | Three strategies with distinct theses: range MR, trend breakout, liquidation fade |
| 1m only, no HTF context | `getCandles` 1m/5m/15m/1h; regime classified from 15m ADX/Choppiness + 1h EMA stack |
| OFI + CVD slope + tape imbalance double-count flow | Dropped raw OFI snapshot. CVD slope replaced by **CVD-vs-price divergence** |
| Single OB snapshot is noise | `bookPersistence` over last 3 OBs (stable=same-sign + within 30%) |
| No cost model | `lib/costs.ts` — every signal must clear `expected_pnl_bps >= 2 × round_trip_cost_bps` |
| Composite linear sum of co-linear features | Each strategy is its own thesis; no composite |
| No per-coin or per-strategy calibration | Bandit allocator sets per-strategy weights from rolling Sharpe |
| Sweep on completed candle is late | Liquidation-fade triggers on 5s liquidation cascades, not on candle close |
| Trailing stop gives back move | Partial TP at 1R + chandelier trail on the runner; range-MR has no trail (TP at VWAP) |
| `risk_per_trade %` ignores volatility | Vol-target sizing: `size = vol_target_usd / (atr_pct × √(hold/1440))` |
| RR=1.8 unrealistic with 1m hit-rates | Each strategy chooses its own RR matched to its hit-rate (range_mr ~1.0R, trend ~runner, liq ~1.5R) |
| No correlation guard beyond hand-curated groups | Portfolio vol budget caps total simultaneous risk regardless of correlation list |
| LLM "strategy_optimizer" curve-fits noise | Replaced by Optuna over the backtester with walk-forward |
| Live-only validation | Full Python backtester + walk-forward + LightGBM gate before deploying |

## How to roll it out (do not skip steps)

1. **Push the new entities to Base44.** `BotConfig`, `StrategyState`, `Trade` (extended), and `Liquidation` need their schemas updated. Existing `Trade` rows remain compatible because all new fields are optional.
2. **Set the new BotConfig.** Start with all three strategies disabled. Set `wallet_address`, `selected_coins`, `is_active=false`.
3. **Record data first.** Run `python research/data_recorder/record_l2.py --coins BTC,ETH,SOL --out data/raw` for at least 2 weeks (a month is better). The backtester is only as good as the data behind it.
4. **Backtest each strategy independently.** Look at `summary.json` — particularly `win_rate`, `avg_r_multiple`, `fees_pct_of_gross`, `sharpe`, `max_drawdown_pct`. If `fees_pct_of_gross > 50` it's not viable; tighten edge filter or move to maker-only.
5. **Walk-forward.** Anything whose OOS Sharpe is <50% of in-sample Sharpe is curve-fit; do not deploy.
6. **Optuna search** only on strategies that survived walk-forward. Use the result as a *starting point*, then re-run walk-forward on the optimized params.
7. **Train the ML gate** (optional, but recommended once you have ≥500 trades). Set `BotConfig.ml.enabled=true` and `model_url` to the published model JSON.
8. **Deploy with one strategy first.** Enable `strategy_range_mr` only. Run for a week with `vol_target_usd_per_trade=10` (very small). Watch live PnL match backtest expectations. Only then turn on the others.
9. **Watch the bandit.** `StrategyState.weight` and `.sharpe` should converge after ~50 trades per strategy. If a strategy's weight floors at `min_weight` and stays there, that strategy is likely dead — disable it and revisit the thesis.

## Known gaps / starting templates

These are intentionally minimal so you can fill them with real data:

- **L2 history in the executor is a single snapshot.** `obHistory` in `StrategyContext` carries one OB. To get 3-snapshot persistence working, cache OBs in a Base44 entity over the last 30s. The persistence function is already in `features.ts`.
- **Liquidation feed.** `tapeCollector` doesn't yet write to the `Liquidation` entity. Extend its WebSocket subscription to include `liquidations` per coin and persist. The schema is already defined.
- **ML gate not yet wired into the executor.** Add the model fetch + calibration lookup to `entry.ts` after `passesEdgeFilter`. The training pipeline is complete.
- **Engine.py emits per-trade features only if you log them.** Extend `engine.py` to record the regime/feature snapshot at signal time so `train_classifier.py` has columns to learn from.

## What "god-level" really means here

Honest read after building it: with all of this in place, an achievable target is Sharpe 1.5–2.5 on a good year, max DD 15–25%, fee-paid as % of gross PnL <30%. That's a real, professional intraday system — not a money printer. The biggest single profitability lever in the rebuild is the cost-aware filter combined with post-only entry on `range_mr`: it converts a strategy that loses to fees into one that earns the maker rebate. The second-biggest lever is HTF gating — the v5 executor took trades against the HTF trend constantly.

If a strategy doesn't survive walk-forward, kill it. Don't tune. Don't argue with the data.
