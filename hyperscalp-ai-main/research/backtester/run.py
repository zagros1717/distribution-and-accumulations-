"""
Entry point for backtesting a single strategy across all recorded coins.

    python research/backtester/run.py --strategy range_mr     --data data/raw --out reports/range_mr
    python research/backtester/run.py --strategy trend_breakout --data data/raw --out reports/trend_breakout
    python research/backtester/run.py --strategy liquidation_fade --data data/raw --out reports/liquidation_fade
"""
import argparse, json
from pathlib import Path
import pandas as pd

from .data_loader import load_all
from .engine import BacktestConfig, simulate
from .cost_model import CostConfig
from .strategies import range_mr_signal, trend_breakout_signal, liquidation_fade_signal
from ..eval.metrics import summary as metrics_summary


STRATEGY_REGISTRY = {
    "range_mr": (range_mr_signal, {
        "min_choppiness": 60, "max_adx_15m": 22,
        "vwap_sigma_entry": 2.0, "vwap_sigma_stop": 3.0,
        "max_funding_skew_bps": 20, "tp_at_vwap": True, "use_post_only": True,
    }),
    "trend_breakout": (trend_breakout_signal, {
        "min_adx_15m": 25, "ema_stack_required": True,
        "breakout_lookback_min": 5, "require_cvd_new_extreme": True,
        "partial_tp_r": 1.0, "partial_tp_fraction": 0.5, "chandelier_atr_mult": 2.5,
    }),
    "liquidation_fade": (liquidation_fade_signal, {
        "min_liquidation_usd_5s": 250_000, "sweep_lookback_min": 5,
        "require_aggressor_flip": True, "rr_target": 1.5, "max_holding_minutes": 8,
    }),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--strategy", required=True, choices=list(STRATEGY_REGISTRY.keys()))
    p.add_argument("--data", required=True)
    p.add_argument("--coins", default="BTC,ETH,SOL")
    p.add_argument("--out", required=True)
    p.add_argument("--no-edge-filter", action="store_true")
    args = p.parse_args()

    coins = [c.strip() for c in args.coins.split(",")]
    coin_data = load_all(Path(args.data), coins)
    fn, cfg = STRATEGY_REGISTRY[args.strategy]

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    all_trades = []
    bt_cfg = BacktestConfig(cost=CostConfig(), enforce_edge_filter=not args.no_edge_filter)
    for coin in coins:
        if coin_data[coin]["bars"].empty:
            print(f"[skip] {coin} — no bars")
            continue
        df = simulate(coin_data, coin, fn, {"cfg": cfg}, bt_cfg)
        if not df.empty:
            df.to_parquet(out / f"trades_{coin}.parquet", index=False)
            all_trades.append(df)
            print(f"[{coin}] {len(df)} trades, net=${df['pnl_usd'].sum():.2f}")

    if not all_trades:
        print("No trades produced. Check data window or relax cost filter.")
        return

    full = pd.concat(all_trades, ignore_index=True)
    full.to_parquet(out / "trades_all.parquet", index=False)
    summary = metrics_summary(full)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
