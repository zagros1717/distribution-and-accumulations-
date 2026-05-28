/**
 * Strategy C — Liquidation Cascade Fade.
 *
 * Thesis: when forced liquidations cascade in one direction, the move
 * overshoots fundamentals because liquidation orders are price-insensitive
 * market orders. As soon as the cascade exhausts, the move snaps back.
 *
 * Trigger:
 *   - large net liquidation USD in a 5s window
 *   - 1m sweep of the session extreme in the same direction as the liquidations
 *   - aggressor flip in the opposite direction within the next few seconds (cascade is over)
 *   - take the opposite side
 *
 * Exit: 1.5R RR, no trailing — quick scalp. Time stop in 8 minutes regardless.
 */
import type { StrategyContext, StrategyResult } from './types.ts';
import { atr, sweepCheck, aggressorFlip, liquidationCascade } from '../lib/features.ts';

export function liquidationFade(ctx: StrategyContext): StrategyResult {
  const cfg = ctx.cfg;
  if (!cfg?.enabled) return { signal: null, reject: 'disabled' };
  if (ctx.c1m.length < 30) return { signal: null, reject: 'insufficient 1m candles' };

  const cascade = liquidationCascade(ctx.liquidations, ctx.coin, 5_000);
  if (Math.abs(cascade.longUsd - cascade.shortUsd) < cfg.min_liquidation_usd_5s) {
    return { signal: null, reject: `cascade $${Math.abs(cascade.longUsd - cascade.shortUsd).toFixed(0)} < min` };
  }
  // Direction: longs liquidating = price was dumping = fade by buying
  const direction: 'long' | 'short' = cascade.longUsd > cascade.shortUsd ? 'long' : 'short';

  const sweep = sweepCheck(ctx.c1m, cfg.sweep_lookback_min ?? 5);
  const sweptCorrectSide = direction === 'long' ? sweep.sweptLow : sweep.sweptHigh;
  if (!sweptCorrectSide) return { signal: null, reject: 'no sweep of session extreme on correct side' };

  if (cfg.require_aggressor_flip) {
    const flip = aggressorFlip(ctx.tape, 10_000, 60_000);
    const flipCorrect = direction === 'long' ? flip > 0.10 : flip < -0.10;
    if (!flipCorrect) return { signal: null, reject: `no aggressor flip (${flip.toFixed(2)})` };
  }

  const last = ctx.c1m[ctx.c1m.length - 1];
  const price = last.c;
  const a1 = atr(ctx.c1m, 14);
  // Stop just past the swept level
  const stop = direction === 'long' ? sweep.prevLow - a1 * 0.3 : sweep.prevHigh + a1 * 0.3;
  const r = Math.abs(price - stop);
  const target = direction === 'long' ? price + r * (cfg.rr_target ?? 1.5) : price - r * (cfg.rr_target ?? 1.5);

  return {
    signal: {
      strategy: 'liquidation_fade',
      direction,
      entry: price,
      stop, target,
      expectedHoldingMinutes: cfg.max_holding_minutes ?? 8,
      preferredEntryMode: 'ioc',
      trailType: 'none',
      reason: `LIQ_FADE ${direction.toUpperCase()} | longLiq=$${cascade.longUsd.toFixed(0)} shortLiq=$${cascade.shortUsd.toFixed(0)}`,
      features: {
        long_liq_usd: cascade.longUsd, short_liq_usd: cascade.shortUsd,
        atr: a1, swept_level: direction === 'long' ? sweep.prevLow : sweep.prevHigh,
      },
    },
  };
}
