"""
Hyperliquid cost model. Mirrors base44/functions/botExecutor/lib/costs.ts so the
backtester sees identical costs to live trading.

Conservative defaults:
    taker fee:        4.5 bps each side
    maker fee:        1.5 bps each side (rebated to 0 on some HL tiers — set explicitly)
    base slippage:     5 bps  (executes inside the spread)
    vol slippage:     15 bps  (high-vol coins drift while you cross)
"""
from dataclasses import dataclass


@dataclass
class CostConfig:
    taker_fee_bps: float = 4.5
    maker_fee_bps: float = 1.5
    slippage_base_bps: float = 5.0
    slippage_vol_bps: float = 15.0
    min_edge_multiplier: float = 2.0


def round_trip_cost_bps(cfg: CostConfig, entry_is_maker: bool, exit_is_maker: bool, realized_vol: float) -> float:
    entry_fee = cfg.maker_fee_bps if entry_is_maker else cfg.taker_fee_bps
    exit_fee = cfg.maker_fee_bps if exit_is_maker else cfg.taker_fee_bps
    slip = cfg.slippage_base_bps + cfg.slippage_vol_bps * min(1.0, realized_vol / 0.01)
    slip_entry = 0.0 if entry_is_maker else slip
    slip_exit = 0.0 if exit_is_maker else slip
    return entry_fee + exit_fee + slip_entry + slip_exit


def expected_pnl_bps(entry: float, target: float) -> float:
    if entry <= 0:
        return 0.0
    return abs(target - entry) / entry * 10000


def passes_edge_filter(cfg: CostConfig, entry: float, target: float, entry_is_maker: bool, exit_is_maker: bool, realized_vol: float):
    cost = round_trip_cost_bps(cfg, entry_is_maker, exit_is_maker, realized_vol)
    pnl = expected_pnl_bps(entry, target)
    return {
        "ok": pnl >= cfg.min_edge_multiplier * cost,
        "cost_bps": cost,
        "pnl_bps": pnl,
        "edge_ratio": pnl / cost if cost > 0 else 0.0,
    }
