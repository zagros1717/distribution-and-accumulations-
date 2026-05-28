/**
 * Common types for all strategies. Each strategy is a pure function:
 *   features -> Signal | null
 *
 * The executor handles sizing, cost-filter, exchange interaction, and bookkeeping.
 */
import type { Candle, OrderBook } from '../lib/hl.ts';
import type { TapeTrade, LiquidationEvent } from '../lib/features.ts';
import type { Regime } from '../lib/regime.ts';

export type StrategyContext = {
  coin: string;
  cfg: any; // BotConfig sub-section for this strategy
  c1m: Candle[];
  c5m: Candle[];
  c15m: Candle[];
  c1h: Candle[];
  ob: OrderBook;
  obHistory: OrderBook[]; // recent OBs for persistence
  tape: TapeTrade[];
  liquidations: LiquidationEvent[];
  funding: number;
  regime: Regime;
  realizedVol: number;
  btcLead: number;
};

export type Signal = {
  strategy: 'range_mr' | 'trend_breakout' | 'liquidation_fade';
  direction: 'long' | 'short';
  entry: number;
  stop: number;
  target: number;
  expectedHoldingMinutes: number;
  preferredEntryMode: 'ioc' | 'post_only' | 'depth_aware';
  partialTpR?: number;
  partialTpFraction?: number;
  trailType?: 'none' | 'chandelier';
  trailAtrMult?: number;
  reason: string;
  features: Record<string, number | string | boolean>;
};

export type StrategyResult = { signal: Signal | null; reject?: string };
