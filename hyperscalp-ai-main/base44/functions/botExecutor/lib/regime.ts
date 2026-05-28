/**
 * Regime classification. A trade idea is only valid in the regime it was designed for.
 *
 * Inputs are HTF candles (15m + 1h) and intraday volatility. Output is one of:
 *   - TREND_UP / TREND_DOWN  -> only trend_breakout strategy can fire
 *   - RANGE                  -> only range_mr can fire
 *   - HIGH_VOL / NEWS        -> only liquidation_fade can fire (and only with cascade)
 *   - NO_TRADE               -> nothing
 */
import type { Candle } from './hl.ts';
import { adx, choppiness, emaStack, atr } from './features.ts';

export type Regime =
  | 'TREND_UP'
  | 'TREND_DOWN'
  | 'RANGE'
  | 'HIGH_VOL'
  | 'NO_TRADE';

export function classifyRegime(c1m: Candle[], c15m: Candle[], c1h: Candle[]): {
  regime: Regime;
  adx15: number;
  chop15: number;
  htfStack: number;
  realizedVol: number;
} {
  const adx15 = adx(c15m, 14);
  const chop15 = choppiness(c15m, 14);
  const { stack: htfStack } = emaStack(c1h);

  // Realized vol = ATR(1m, 14) / price
  const a1 = atr(c1m, 14);
  const lastPx = c1m[c1m.length - 1]?.c ?? 0;
  const realizedVol = lastPx > 0 ? a1 / lastPx : 0;

  // Hard NO_TRADE on extreme realized vol unless explicitly fading liquidations
  // (the strategy itself decides what to do with HIGH_VOL).
  if (realizedVol > 0.012) return { regime: 'HIGH_VOL', adx15, chop15, htfStack, realizedVol };

  // Trending (only when both ADX confirms and HTF agrees)
  if (adx15 >= 25 && htfStack === 1) return { regime: 'TREND_UP', adx15, chop15, htfStack, realizedVol };
  if (adx15 >= 25 && htfStack === -1) return { regime: 'TREND_DOWN', adx15, chop15, htfStack, realizedVol };

  // Ranging
  if (adx15 < 22 && chop15 > 60) return { regime: 'RANGE', adx15, chop15, htfStack, realizedVol };

  return { regime: 'NO_TRADE', adx15, chop15, htfStack, realizedVol };
}
