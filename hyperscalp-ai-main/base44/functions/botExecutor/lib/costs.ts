/**
 * Cost model — every trade has to clear the cost-aware filter or it's skipped.
 *
 *   round_trip_cost_bps = entry_fee + exit_fee + entry_slip + exit_slip
 *   expected_pnl_bps    = abs(tp - entry) / entry * 10000   (gross, before fees)
 *
 *   Trade is allowed iff expected_pnl_bps >= min_edge_multiplier * round_trip_cost_bps.
 *
 * v5 had no cost model at all — that's the single most expensive bug for taker scalping.
 */
export type CostConfig = {
  taker_fee_bps: number;
  maker_fee_bps: number;
  slippage_base_bps: number;
  slippage_vol_bps: number;
  min_edge_multiplier: number;
};

export function roundTripCostBps(cfg: CostConfig, entryIsMaker: boolean, exitIsMaker: boolean, realizedVol: number): number {
  const entryFee = entryIsMaker ? cfg.maker_fee_bps : cfg.taker_fee_bps;
  const exitFee = exitIsMaker ? cfg.maker_fee_bps : cfg.taker_fee_bps;
  // Slippage scales with realized vol (a coin moving 5%/day will slip more than one moving 1%)
  const slip = cfg.slippage_base_bps + cfg.slippage_vol_bps * Math.min(1, realizedVol / 0.01);
  const slipPerSide = entryIsMaker ? 0 : slip; // makers don't slip the book on entry
  const slipExit = exitIsMaker ? 0 : slip;
  return entryFee + exitFee + slipPerSide + slipExit;
}

export function expectedPnlBps(entry: number, target: number): number {
  if (entry <= 0) return 0;
  return Math.abs(target - entry) / entry * 10000;
}

export function passesEdgeFilter(cfg: CostConfig, entry: number, target: number, entryIsMaker: boolean, exitIsMaker: boolean, realizedVol: number) {
  const cost = roundTripCostBps(cfg, entryIsMaker, exitIsMaker, realizedVol);
  const pnl = expectedPnlBps(entry, target);
  const ok = pnl >= cfg.min_edge_multiplier * cost;
  return { ok, costBps: cost, pnlBps: pnl, edgeRatio: cost > 0 ? pnl / cost : 0 };
}
