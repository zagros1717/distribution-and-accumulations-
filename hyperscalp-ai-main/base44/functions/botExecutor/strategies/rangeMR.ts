/**
 * Strategy A — Range Mean Reversion.
 *
 * Thesis: in a chopping market, price oscillates around session VWAP. Fade
 * 2σ deviations when:
 *   - regime is RANGE (ADX low, choppiness high)
 *   - tape shows absorption (large prints into the level that don't move price)
 *   - funding skew isn't strongly against us
 *   - book imbalance is consistent on the *correct* side (bids stacked when fading lows, asks when fading highs)
 *
 * Exit:
 *   - TP at VWAP (the "mean")
 *   - SL outside the range (3σ from VWAP)
 *   - prefer post-only maker entry → maker fee + fill at our price
 */
import type { StrategyContext, StrategyResult } from './types.ts';
import { vwapBands, atr, bookPersistence, largePrints, aggressorFlip } from '../lib/features.ts';

export function rangeMR(ctx: StrategyContext): StrategyResult {
  const cfg = ctx.cfg;
  if (!cfg?.enabled) return { signal: null, reject: 'disabled' };
  if (ctx.regime !== 'RANGE') return { signal: null, reject: `regime=${ctx.regime}` };
  if (ctx.c1m.length < 60) return { signal: null, reject: 'insufficient 1m candles' };

  // Use last 60 1m bars as the "session" for VWAP computation
  const session = ctx.c1m.slice(-60);
  const { vwap, sigma, lastDeviation } = vwapBands(session);
  if (sigma === 0) return { signal: null, reject: 'zero VWAP sigma' };

  const last = ctx.c1m[ctx.c1m.length - 1];
  const price = last.c;

  // Direction is opposite the deviation
  let direction: 'long' | 'short' | null = null;
  if (lastDeviation <= -cfg.vwap_sigma_entry) direction = 'long';
  if (lastDeviation >= cfg.vwap_sigma_entry) direction = 'short';
  if (!direction) return { signal: null, reject: `dev=${lastDeviation.toFixed(2)}σ < entry=${cfg.vwap_sigma_entry}σ` };

  // Funding skew: if longs are paying heavily, fading rallies is good but fading dips is bad
  const fundingBps = ctx.funding * 10000;
  if (direction === 'long' && fundingBps > cfg.max_funding_skew_bps) return { signal: null, reject: `funding=${fundingBps.toFixed(1)}bps too long-loaded` };
  if (direction === 'short' && fundingBps < -cfg.max_funding_skew_bps) return { signal: null, reject: `funding=${fundingBps.toFixed(1)}bps too short-loaded` };

  // Absorption: large prints into the level. For a long fade at the low, we want
  // big aggressor sells that didn't push price meaningfully lower (price holds within ATR).
  const a1 = atr(ctx.c1m, 14);
  const lp = largePrints(ctx.tape, 90_000, 25_000);
  const absorptionOk = direction === 'long'
    ? lp.sells >= 2 && (last.c - last.l) >= a1 * 0.3 // small lower wick → buyers stepped in
    : lp.buys >= 2 && (last.h - last.c) >= a1 * 0.3;
  if (!absorptionOk) return { signal: null, reject: `no absorption (${direction === 'long' ? 'sell' : 'buy'}LP=${direction === 'long' ? lp.sells : lp.buys})` };

  // Aggressor flip in our direction (recent shift from sells->buys for longs)
  const flip = aggressorFlip(ctx.tape);
  if (direction === 'long' && flip < 0.05) return { signal: null, reject: `weak bullish flip ${flip.toFixed(2)}` };
  if (direction === 'short' && flip > -0.05) return { signal: null, reject: `weak bearish flip ${flip.toFixed(2)}` };

  // Book persistence — order book imbalance must agree
  const { meanImb, stable } = bookPersistence(ctx.obHistory.slice(-3));
  if (stable) {
    if (direction === 'long' && meanImb < -0.10) return { signal: null, reject: `book persistently asks-heavy ${meanImb.toFixed(2)}` };
    if (direction === 'short' && meanImb > 0.10) return { signal: null, reject: `book persistently bids-heavy ${meanImb.toFixed(2)}` };
  }

  // BTC leadership filter: don't fight a BTC dump on alt-longs
  if (direction === 'long' && ctx.btcLead < 0 && ctx.coin !== 'BTC') return { signal: null, reject: 'BTC dumping' };
  if (direction === 'short' && ctx.btcLead > 0 && ctx.coin !== 'BTC') return { signal: null, reject: 'BTC pumping' };

  // Targets
  const target = cfg.tp_at_vwap ? vwap : (direction === 'long' ? price + sigma : price - sigma);
  const stop = direction === 'long' ? price - sigma * cfg.vwap_sigma_stop : price + sigma * cfg.vwap_sigma_stop;

  return {
    signal: {
      strategy: 'range_mr',
      direction,
      entry: price,
      stop, target,
      expectedHoldingMinutes: 12,
      preferredEntryMode: cfg.use_post_only ? 'post_only' : 'depth_aware',
      trailType: 'none',
      reason: `RANGE_MR ${direction.toUpperCase()} dev=${lastDeviation.toFixed(2)}σ vwap=${vwap.toFixed(4)} flip=${flip.toFixed(2)}`,
      features: {
        vwap, sigma, deviation_sigma: lastDeviation, atr: a1,
        large_buys: lp.buys, large_sells: lp.sells, aggressor_flip: flip,
        book_imb: meanImb, funding_bps: fundingBps,
      },
    },
  };
}
