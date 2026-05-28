/**
 * Strategy B — Trend Continuation Breakout.
 *
 * Thesis: in an established HTF trend (1h EMA stack + 15m ADX>25), pullbacks
 * resolve in the trend's direction. Enter on a 1m close that breaks the prior
 * 5m high (or low for shorts), but ONLY when CVD also makes a new extreme
 * (real divergence check, not slope). This filters out fake breakouts where
 * price ticks higher but aggressor flow doesn't follow.
 *
 * Exit: partial TP at 1R, chandelier trail on the runner.
 */
import type { StrategyContext, StrategyResult } from './types.ts';
import { atr, cvdSeries } from '../lib/features.ts';

export function trendBreakout(ctx: StrategyContext): StrategyResult {
  const cfg = ctx.cfg;
  if (!cfg?.enabled) return { signal: null, reject: 'disabled' };
  if (ctx.regime !== 'TREND_UP' && ctx.regime !== 'TREND_DOWN') return { signal: null, reject: `regime=${ctx.regime}` };
  if (ctx.c1m.length < 60) return { signal: null, reject: 'insufficient 1m candles' };

  const direction: 'long' | 'short' = ctx.regime === 'TREND_UP' ? 'long' : 'short';

  const lookback = cfg.breakout_lookback_min ?? 5;
  const recent = ctx.c1m.slice(-(lookback + 1), -1);
  if (recent.length < lookback) return { signal: null, reject: 'breakout window too small' };
  const recentHigh = Math.max(...recent.map(c => c.h));
  const recentLow = Math.min(...recent.map(c => c.l));

  const last = ctx.c1m[ctx.c1m.length - 1];
  const price = last.c;

  // Breakout check on the just-closed bar
  const broke = direction === 'long' ? last.c > recentHigh : last.c < recentLow;
  if (!broke) return { signal: null, reject: `no breakout (price=${price} vs ${direction === 'long' ? recentHigh : recentLow})` };

  // CVD must also make a new extreme over the same window — this is the real divergence check
  if (cfg.require_cvd_new_extreme) {
    const { cvd } = cvdSeries(ctx.tape, (lookback + 2) * 60_000);
    if (cvd.length < 4) return { signal: null, reject: 'insufficient CVD samples' };
    const lastCvd = cvd[cvd.length - 1];
    const priorCvd = cvd.slice(0, -1);
    const cvdNewExtreme = direction === 'long'
      ? lastCvd >= Math.max(...priorCvd)
      : lastCvd <= Math.min(...priorCvd);
    if (!cvdNewExtreme) return { signal: null, reject: 'price broke but CVD did not (fake breakout)' };
  }

  // BTC leadership: alts shouldn't break out alone if BTC isn't moving with us
  if (ctx.coin !== 'BTC') {
    if (direction === 'long' && ctx.btcLead < 0) return { signal: null, reject: 'BTC dumping; alt long unsafe' };
    if (direction === 'short' && ctx.btcLead > 0) return { signal: null, reject: 'BTC pumping; alt short unsafe' };
  }

  const a1 = atr(ctx.c1m, 14);
  // Stop just outside the breakout structure (other side of recent range)
  const stop = direction === 'long' ? recentLow - a1 * 0.2 : recentHigh + a1 * 0.2;
  const r = Math.abs(price - stop);
  // Initial target for cost-filter purposes is 1R (partial TP). The runner trails.
  const target = direction === 'long' ? price + r * 1.0 : price - r * 1.0;

  return {
    signal: {
      strategy: 'trend_breakout',
      direction,
      entry: price,
      stop, target,
      expectedHoldingMinutes: 18,
      preferredEntryMode: 'depth_aware',
      partialTpR: cfg.partial_tp_r ?? 1.0,
      partialTpFraction: cfg.partial_tp_fraction ?? 0.5,
      trailType: 'chandelier',
      trailAtrMult: cfg.chandelier_atr_mult ?? 2.5,
      reason: `TREND ${direction.toUpperCase()} | brk=${(direction === 'long' ? recentHigh : recentLow).toFixed(4)} | adx=${'computed in regime'}`,
      features: {
        breakout_level: direction === 'long' ? recentHigh : recentLow,
        atr: a1, r_distance: r,
      },
    },
  };
}
