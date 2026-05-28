/**
 * Multi-armed bandit allocator across the three strategies.
 *
 * - Each strategy has rolling state in StrategyState: ret_window (R-multiples),
 *   trades_window, kelly_fraction, sharpe.
 * - Allocator computes weights using a softmax over Sharpe with epsilon-greedy
 *   exploration so a bad-streak strategy still gets some capital.
 * - Final weights are renormalized over enabled strategies and floored at min_weight.
 */
export type StrategyName = 'range_mr' | 'trend_breakout' | 'liquidation_fade';

export type StrategyStats = {
  strategy: StrategyName;
  trades_window: number;
  sharpe: number;
  kelly_fraction: number;
  weight: number;
};

export type AllocatorConfig = {
  use_bandit: boolean;
  exploration_eps: number;
  min_weight: number;
};

export function computeWeights(stats: StrategyStats[], enabled: Record<StrategyName, boolean>, cfg: AllocatorConfig): Record<StrategyName, number> {
  const live = stats.filter(s => enabled[s.strategy]);
  if (live.length === 0) return { range_mr: 0, trend_breakout: 0, liquidation_fade: 0 } as any;

  if (!cfg.use_bandit || live.every(s => s.trades_window < 10)) {
    // Equal split until any strategy has enough trades to evaluate
    const w = 1 / live.length;
    const out: any = { range_mr: 0, trend_breakout: 0, liquidation_fade: 0 };
    for (const s of live) out[s.strategy] = w;
    return out;
  }

  // Softmax over Sharpe — temperature 1.0
  const sharpes = live.map(s => s.sharpe);
  const maxS = Math.max(...sharpes);
  const exps = sharpes.map(x => Math.exp(x - maxS));
  const sumExp = exps.reduce((a, b) => a + b, 0);
  const softmax = exps.map(e => e / sumExp);

  // Epsilon-greedy: blend a uniform component
  const eps = cfg.exploration_eps;
  const blended = softmax.map(w => (1 - eps) * w + eps / live.length);

  // Floor at min_weight then renormalize
  const floored = blended.map(w => Math.max(cfg.min_weight, w));
  const sum = floored.reduce((a, b) => a + b, 0);
  const normalized = floored.map(w => w / sum);

  const out: any = { range_mr: 0, trend_breakout: 0, liquidation_fade: 0 };
  live.forEach((s, i) => { out[s.strategy] = normalized[i]; });
  return out;
}
