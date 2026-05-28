/**
 * Volatility-targeted sizing + per-strategy Kelly cap + portfolio vol budget.
 *
 * Replaces the old fixed `risk_per_trade %`. The intuition: every trade should
 * contribute roughly the same daily-vol $ to the portfolio, regardless of which
 * coin it's on or how wide the stop is.
 *
 * size_usd = vol_target_usd / (atr_pct * sqrt(holding_minutes / 1440))
 *
 * Then capped by:
 *   - per-trade max (hard cap by exchange)
 *   - max_leverage * equity / N_open_slots
 *   - remaining portfolio vol budget
 *   - kelly_cap fraction of strategy_state.kelly_fraction
 */
export type SizingInputs = {
  equity: number;
  vol_target_usd_per_trade: number;
  portfolio_vol_budget_usd: number;
  current_portfolio_vol_usd: number;
  max_leverage: number;
  atr_pct: number; // ATR(1m) / price
  expected_holding_minutes: number;
  per_trade_min_usd: number;
  per_trade_max_usd: number;
  strategy_kelly_fraction?: number; // 0..1, optional from StrategyState
  allocator_kelly_cap?: number; // typically 0.25
  bandit_weight?: number; // 0..1, optional from allocator
};

export function calculateSize(i: SizingInputs): { sizeUsd: number; reason: string } {
  if (i.atr_pct <= 0 || i.equity <= 0) return { sizeUsd: 0, reason: 'invalid inputs' };

  const dayFraction = Math.max(1 / 1440, i.expected_holding_minutes / 1440);
  const expectedMove = i.atr_pct * Math.sqrt(dayFraction); // expected % move during hold
  const baseSize = expectedMove > 0 ? i.vol_target_usd_per_trade / expectedMove : 0;

  // Apply Kelly fraction × cap (only scales DOWN)
  const kelly = Math.min(1, Math.max(0, (i.strategy_kelly_fraction ?? 1) * (i.allocator_kelly_cap ?? 1)));
  const banditScaled = baseSize * kelly * (i.bandit_weight ?? 1);

  // Remaining vol budget — if portfolio is full, scale toward zero
  const remainingBudget = Math.max(0, i.portfolio_vol_budget_usd - i.current_portfolio_vol_usd);
  const budgetScale = i.portfolio_vol_budget_usd > 0 ? Math.min(1, remainingBudget / i.vol_target_usd_per_trade) : 1;
  const budgetScaled = banditScaled * budgetScale;

  // Leverage cap
  const levCap = i.equity * i.max_leverage;
  const sizeBeforeBounds = Math.min(budgetScaled, levCap);

  // Hard min/max
  const sizeUsd = Math.max(0, Math.min(i.per_trade_max_usd, sizeBeforeBounds));
  if (sizeUsd < i.per_trade_min_usd) return { sizeUsd: 0, reason: `size $${sizeUsd.toFixed(2)} < min $${i.per_trade_min_usd}` };

  return {
    sizeUsd: parseFloat(sizeUsd.toFixed(2)),
    reason: `voltgt=$${i.vol_target_usd_per_trade} kelly=${kelly.toFixed(2)} bandit=${(i.bandit_weight ?? 1).toFixed(2)} budget=${budgetScale.toFixed(2)}`,
  };
}

/**
 * Kelly fraction from rolling per-trade R-multiple returns.
 * f* = mean(R) / variance(R)  —  capped to [0, 1]. Returns 0 if not enough history.
 */
export function kellyFraction(rMultiples: number[], minTrades = 20): number {
  if (rMultiples.length < minTrades) return 0;
  const n = rMultiples.length;
  const mean = rMultiples.reduce((a, b) => a + b, 0) / n;
  const variance = rMultiples.reduce((a, b) => a + (b - mean) ** 2, 0) / n;
  if (variance <= 0) return 0;
  return Math.max(0, Math.min(1, mean / variance));
}

/** Annualized Sharpe approximation from per-trade R returns + trades-per-day estimate. */
export function rollingSharpe(rMultiples: number[], tradesPerDay = 10): number {
  if (rMultiples.length < 10) return 0;
  const n = rMultiples.length;
  const mean = rMultiples.reduce((a, b) => a + b, 0) / n;
  const variance = rMultiples.reduce((a, b) => a + (b - mean) ** 2, 0) / n;
  const std = Math.sqrt(variance);
  if (std === 0) return 0;
  return (mean / std) * Math.sqrt(tradesPerDay * 252);
}
