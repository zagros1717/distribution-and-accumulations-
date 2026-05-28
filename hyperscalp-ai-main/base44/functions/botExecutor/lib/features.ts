/**
 * Feature pipeline. Pure functions over candles / order book / tape /
 * higher-timeframe candles / OI / funding / liquidations.
 *
 * Real edge features only. The v5 alphas (raw OFI snapshot, raw CVD slope)
 * are deliberately dropped — they were noisy and double-counted. New:
 *   - HTF EMA stack + slope (1h, 15m)
 *   - ADX, Choppiness Index (range vs trend)
 *   - VWAP + standard-deviation bands
 *   - CVD-vs-price divergence (the actual edge from CVD)
 *   - Book-imbalance persistence (3-snapshot rolling)
 *   - Large-print count from tape
 *   - Liquidation cascade detection
 *   - Realized vol vs ATR
 */
import type { Candle, OrderBook } from './hl.ts';

export type TapeTrade = { t: number; p: number; s: number; b: boolean };

// ─── Basic indicators ────────────────────────────────────────────────────
export function ema(values: number[], period: number): number[] {
  if (values.length === 0) return [];
  const k = 2 / (period + 1);
  const out = [values[0]];
  for (let i = 1; i < values.length; i++) out.push(values[i] * k + out[i - 1] * (1 - k));
  return out;
}

export function atr(candles: Candle[], period = 14): number {
  if (candles.length < period + 1) return 0;
  const trs: number[] = [];
  for (let i = 1; i < candles.length; i++) {
    const h = candles[i].h, l = candles[i].l, pc = candles[i - 1].c;
    trs.push(Math.max(h - l, Math.abs(h - pc), Math.abs(l - pc)));
  }
  return trs.slice(-period).reduce((a, b) => a + b, 0) / period;
}

/**
 * ADX (Wilder) — measures trend strength. >25 trend, <20 chop.
 */
export function adx(candles: Candle[], period = 14): number {
  if (candles.length < period * 2) return 0;
  const tr: number[] = []; const plusDM: number[] = []; const minusDM: number[] = [];
  for (let i = 1; i < candles.length; i++) {
    const up = candles[i].h - candles[i - 1].h;
    const dn = candles[i - 1].l - candles[i].l;
    plusDM.push(up > dn && up > 0 ? up : 0);
    minusDM.push(dn > up && dn > 0 ? dn : 0);
    const h = candles[i].h, l = candles[i].l, pc = candles[i - 1].c;
    tr.push(Math.max(h - l, Math.abs(h - pc), Math.abs(l - pc)));
  }
  // Wilder smoothing
  const wsmooth = (arr: number[], p: number) => {
    let s = arr.slice(0, p).reduce((a, b) => a + b, 0);
    const out = [s];
    for (let i = p; i < arr.length; i++) {
      s = s - s / p + arr[i];
      out.push(s);
    }
    return out;
  };
  const trS = wsmooth(tr, period);
  const pdmS = wsmooth(plusDM, period);
  const mdmS = wsmooth(minusDM, period);
  const dx: number[] = [];
  for (let i = 0; i < trS.length; i++) {
    if (trS[i] === 0) { dx.push(0); continue; }
    const pdi = 100 * pdmS[i] / trS[i];
    const mdi = 100 * mdmS[i] / trS[i];
    const sum = pdi + mdi;
    dx.push(sum === 0 ? 0 : 100 * Math.abs(pdi - mdi) / sum);
  }
  if (dx.length < period) return 0;
  // Average DX over `period` to get ADX
  const recent = dx.slice(-period);
  return recent.reduce((a, b) => a + b, 0) / recent.length;
}

/**
 * Choppiness Index (Dreiss). High = chop, low = trend. Range ~0..100.
 */
export function choppiness(candles: Candle[], period = 14): number {
  if (candles.length < period + 1) return 0;
  const recent = candles.slice(-period);
  let trSum = 0;
  for (let i = 1; i < recent.length; i++) {
    const h = recent[i].h, l = recent[i].l, pc = recent[i - 1].c;
    trSum += Math.max(h - l, Math.abs(h - pc), Math.abs(l - pc));
  }
  const hi = Math.max(...recent.map(c => c.h));
  const lo = Math.min(...recent.map(c => c.l));
  if (hi - lo === 0 || trSum === 0) return 0;
  return 100 * Math.log10(trSum / (hi - lo)) / Math.log10(period);
}

/**
 * Session VWAP + standard deviation bands. Session = the candles passed in.
 */
export function vwapBands(candles: Candle[]): { vwap: number; sigma: number; lastDeviation: number } {
  if (candles.length === 0) return { vwap: 0, sigma: 0, lastDeviation: 0 };
  let pvSum = 0, vSum = 0;
  for (const c of candles) {
    const tp = (c.h + c.l + c.c) / 3;
    pvSum += tp * c.v; vSum += c.v;
  }
  const vwap = vSum > 0 ? pvSum / vSum : candles[candles.length - 1].c;
  // Volume-weighted variance of typical price
  let varNum = 0;
  for (const c of candles) {
    const tp = (c.h + c.l + c.c) / 3;
    varNum += c.v * (tp - vwap) ** 2;
  }
  const sigma = vSum > 0 ? Math.sqrt(varNum / vSum) : 0;
  const lastClose = candles[candles.length - 1].c;
  const lastDeviation = sigma > 0 ? (lastClose - vwap) / sigma : 0;
  return { vwap, sigma, lastDeviation };
}

/**
 * EMA stack on HTF candles. Returns +1 if EMA20>EMA50>EMA100, -1 if reversed, 0 otherwise.
 */
export function emaStack(candles: Candle[]): { stack: number; slope20: number } {
  if (candles.length < 100) return { stack: 0, slope20: 0 };
  const closes = candles.map(c => c.c);
  const e20 = ema(closes, 20);
  const e50 = ema(closes, 50);
  const e100 = ema(closes, 100);
  const last = (a: number[]) => a[a.length - 1];
  const stack = last(e20) > last(e50) && last(e50) > last(e100) ? 1
              : last(e20) < last(e50) && last(e50) < last(e100) ? -1 : 0;
  // Slope of EMA20 normalized by price
  const slope20 = (last(e20) - e20[e20.length - 5]) / last(e20);
  return { stack, slope20 };
}

// ─── Tape-derived features (real CVD divergence + large prints) ──────────
export function cvdSeries(tape: TapeTrade[], windowMs: number, buckets = 12): { cvd: number[]; price: number[]; ts: number[] } {
  if (tape.length === 0) return { cvd: [], price: [], ts: [] };
  const now = Date.now();
  const start = now - windowMs;
  const recent = tape.filter(t => t.t >= start).sort((a, b) => a.t - b.t);
  if (recent.length < buckets) return { cvd: [], price: [], ts: [] };
  const bucketMs = windowMs / buckets;
  const cvd: number[] = []; const price: number[] = []; const ts: number[] = [];
  let cum = 0; let bucketStart = start;
  for (let i = 0; i < buckets; i++) {
    const bucketEnd = bucketStart + bucketMs;
    let lastPx = 0;
    for (const tr of recent) {
      if (tr.t < bucketStart || tr.t >= bucketEnd) continue;
      cum += tr.b ? tr.s : -tr.s;
      lastPx = tr.p;
    }
    cvd.push(cum);
    price.push(lastPx || (price[price.length - 1] ?? recent[0].p));
    ts.push(bucketEnd);
    bucketStart = bucketEnd;
  }
  return { cvd, price, ts };
}

/**
 * CVD-vs-price divergence over a window. Returns:
 *   +1 = bullish divergence (price made new low, CVD made higher low) — long signal
 *   -1 = bearish divergence (price made new high, CVD made lower high) — short signal
 *    0 = no divergence
 * Uses the bucketed cvd/price series so both can be checked for new highs/lows.
 */
export function cvdDivergence(tape: TapeTrade[], windowMs = 8 * 60 * 1000): { sign: number; strength: number } {
  const { cvd, price } = cvdSeries(tape, windowMs);
  if (cvd.length < 6) return { sign: 0, strength: 0 };
  const half = Math.floor(cvd.length / 2);
  const priceHi1 = Math.max(...price.slice(0, half));
  const priceHi2 = Math.max(...price.slice(half));
  const priceLo1 = Math.min(...price.slice(0, half));
  const priceLo2 = Math.min(...price.slice(half));
  const cvdHi1 = Math.max(...cvd.slice(0, half));
  const cvdHi2 = Math.max(...cvd.slice(half));
  const cvdLo1 = Math.min(...cvd.slice(0, half));
  const cvdLo2 = Math.min(...cvd.slice(half));
  if (priceHi2 > priceHi1 && cvdHi2 < cvdHi1) {
    const strength = (priceHi2 - priceHi1) / Math.max(priceHi1, 1) + (cvdHi1 - cvdHi2) / Math.max(Math.abs(cvdHi1), 1);
    return { sign: -1, strength: Math.min(1, strength) };
  }
  if (priceLo2 < priceLo1 && cvdLo2 > cvdLo1) {
    const strength = (priceLo1 - priceLo2) / Math.max(priceLo1, 1) + (cvdLo2 - cvdLo1) / Math.max(Math.abs(cvdLo1), 1);
    return { sign: 1, strength: Math.min(1, strength) };
  }
  return { sign: 0, strength: 0 };
}

/**
 * Count of large aggressor prints in a window — proxy for "smart money" activity.
 */
export function largePrints(tape: TapeTrade[], windowMs: number, minUsd: number): { buys: number; sells: number; netUsd: number } {
  const start = Date.now() - windowMs;
  let buys = 0, sells = 0, netUsd = 0;
  for (const tr of tape) {
    if (tr.t < start) continue;
    const usd = tr.p * tr.s;
    if (usd < minUsd) continue;
    if (tr.b) { buys++; netUsd += usd; } else { sells++; netUsd -= usd; }
  }
  return { buys, sells, netUsd };
}

/**
 * Recent aggressor flip: short-window (e.g. last 15s) tape imbalance vs the
 * minute prior. Used by the liquidation-fade strategy to confirm a flush is over.
 */
export function aggressorFlip(tape: TapeTrade[], shortMs = 15_000, longMs = 60_000): number {
  const now = Date.now();
  let shortB = 0, shortS = 0, longB = 0, longS = 0;
  for (const tr of tape) {
    if (now - tr.t > longMs) continue;
    if (tr.b) { longB += tr.s; if (now - tr.t <= shortMs) shortB += tr.s; }
    else { longS += tr.s; if (now - tr.t <= shortMs) shortS += tr.s; }
  }
  const sTot = shortB + shortS; const lTot = longB + longS;
  const sImb = sTot > 0 ? (shortB - shortS) / sTot : 0;
  const lImb = lTot > 0 ? (longB - longS) / lTot : 0;
  return sImb - lImb; // positive = recent buys after sells (bullish flip)
}

// ─── Order book features ─────────────────────────────────────────────────
export function bookImbalance(ob: OrderBook, levels = 10): number {
  const bids = ob.bids.slice(0, levels).reduce((s, l) => s + l.sz, 0);
  const asks = ob.asks.slice(0, levels).reduce((s, l) => s + l.sz, 0);
  const tot = bids + asks;
  return tot > 0 ? (bids - asks) / tot : 0;
}

/**
 * Persistence: pass in last 3 OBs (oldest first). Returns mean imbalance plus
 * a "stable" flag if all 3 have the same sign and within 30% of each other.
 */
export function bookPersistence(obs: OrderBook[]): { meanImb: number; stable: boolean } {
  if (obs.length < 3) return { meanImb: 0, stable: false };
  const imbs = obs.map(o => bookImbalance(o, 10));
  const mean = imbs.reduce((a, b) => a + b, 0) / imbs.length;
  const sameSign = imbs.every(i => Math.sign(i) === Math.sign(mean));
  const ratios = imbs.map(i => Math.abs(mean) > 0 ? Math.abs(i) / Math.abs(mean) : 0);
  const stable = sameSign && ratios.every(r => r > 0.7 && r < 1.3);
  return { meanImb: mean, stable };
}

// ─── Liquidation cascade detection ───────────────────────────────────────
export type LiquidationEvent = { t: number; coin: string; side: 'long' | 'short'; usd: number };

export function liquidationCascade(events: LiquidationEvent[], coin: string, windowMs = 5_000): { net: number; longUsd: number; shortUsd: number } {
  const now = Date.now();
  let longUsd = 0, shortUsd = 0;
  for (const e of events) {
    if (e.coin !== coin || now - e.t > windowMs) continue;
    if (e.side === 'long') longUsd += e.usd; else shortUsd += e.usd;
  }
  return { net: longUsd - shortUsd, longUsd, shortUsd };
}

// ─── Sweep detection (kept from v5, fixed candle slicing preserved) ──────
export function sweepCheck(candles: Candle[], lookback = 20): { sweptHigh: boolean; sweptLow: boolean; reclaimedHigh: boolean; reclaimedLow: boolean; prevHigh: number; prevLow: number } {
  if (candles.length < lookback + 1) return { sweptHigh: false, sweptLow: false, reclaimedHigh: false, reclaimedLow: false, prevHigh: 0, prevLow: 0 };
  const completed = candles.slice(-(lookback + 1), -1);
  const prevHigh = Math.max(...completed.map(c => c.h));
  const prevLow = Math.min(...completed.map(c => c.l));
  const cur = candles[candles.length - 1];
  return {
    sweptHigh: cur.h > prevHigh,
    sweptLow: cur.l < prevLow,
    reclaimedHigh: cur.h > prevHigh && cur.c < prevHigh,
    reclaimedLow: cur.l < prevLow && cur.c > prevLow,
    prevHigh, prevLow,
  };
}

// ─── BTC leadership filter (basic) ───────────────────────────────────────
/** -1 if BTC is dumping >0.3% in last 5m, +1 if pumping, 0 otherwise. */
export function btcLeadership(btcCandles: Candle[]): number {
  if (btcCandles.length < 6) return 0;
  const now = btcCandles[btcCandles.length - 1].c;
  const ago = btcCandles[btcCandles.length - 6].c;
  const ret = (now - ago) / ago;
  if (ret < -0.003) return -1;
  if (ret > 0.003) return 1;
  return 0;
}
