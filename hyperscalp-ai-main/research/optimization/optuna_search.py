"""
Optuna search over backtester hyperparameters. Optimizes for OOS Sharpe with a
penalty for low trade count (drawdown is implicitly penalized by Sharpe denom).

    python research/optimization/optuna_search.py --strategy range_mr --trials 200 --data data/raw

Recommended: don't trust any winner with <200 OOS trades. Curve-fitting risk explodes below that.
"""
import argparse, json
from pathlib import Path
import optuna
import numpy as np
import pandas as pd

from ..backtester.data_loader import load_all
from ..backtester.engine import BacktestConfig, simulate
from ..backtester.cost_model import CostConfig
from ..backtester.strategies import range_mr_signal, trend_breakout_signal, liquidation_fade_signal


SUGGESTORS = {
    "range_mr": (range_mr_signal, lambda t: {
        "min_choppiness": t.suggest_int("min_choppiness", 50, 70),
        "max_adx_15m": t.suggest_int("max_adx_15m", 18, 28),
        "vwap_sigma_entry": t.suggest_float("vwap_sigma_entry", 1.5, 2.6),
        "vwap_sigma_stop": t.suggest_float("vwap_sigma_stop", 2.0, 4.0),
        "max_funding_skew_bps": t.suggest_int("max_funding_skew_bps", 10, 40),
        "tp_at_vwap": True, "use_post_only": True,
    }),
    "trend_breakout": (trend_breakout_signal, lambda t: {
        "min_adx_15m": t.suggest_int("min_adx_15m", 20, 32),
        "ema_stack_required": True,
        "breakout_lookback_min": t.suggest_int("breakout_lookback_min", 3, 8),
        "require_cvd_new_extreme": t.suggest_categorical("require_cvd_new_extreme", [True, False]),
        "partial_tp_r": t.suggest_float("partial_tp_r", 0.7, 1.5),
        "partial_tp_fraction": t.suggest_float("partial_tp_fraction", 0.3, 0.7),
        "chandelier_atr_mult": t.suggest_float("chandelier_atr_mult", 1.8, 3.5),
    }),
    "liquidation_fade": (liquidation_fade_signal, lambda t: {
        "min_liquidation_usd_5s": t.suggest_int("min_liq_usd", 100_000, 600_000, step=50_000),
        "sweep_lookback_min": t.suggest_int("sweep_lookback_min", 3, 8),
        "require_aggressor_flip": True,
        "rr_target": t.suggest_float("rr_target", 1.2, 2.2),
        "max_holding_minutes": t.suggest_int("max_holding_minutes", 5, 12),
    }),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--strategy", required=True, choices=list(SUGGESTORS.keys()))
    p.add_argument("--data", required=True)
    p.add_argument("--coins", default="BTC,ETH,SOL")
    p.add_argument("--trials", type=int, default=100)
    p.add_argument("--out", default="reports/optuna")
    args = p.parse_args()

    coins = [c.strip() for c in args.coins.split(",")]
    data = load_all(Path(args.data), coins)
    fn, suggest = SUGGESTORS[args.strategy]
    bt_cfg = BacktestConfig(cost=CostConfig())

    def objective(trial):
        cfg = suggest(trial)
        all_trades = []
        for coin in coins:
            if data[coin]["bars"].empty:
                continue
            df = simulate(data, coin, fn, {"cfg": cfg}, bt_cfg)
            if not df.empty:
                all_trades.append(df)
        if not all_trades:
            return -10.0
        full = pd.concat(all_trades, ignore_index=True)
        if len(full) < 30:
            return -5.0
        r = full["r_multiple"]
        sharpe = float(r.mean() / r.std() * np.sqrt(252 * 10)) if r.std() > 0 else 0
        # Penalize low trade count
        penalty = max(0, 100 - len(full)) / 100
        return sharpe - penalty

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=args.trials, show_progress_bar=True)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / f"best_{args.strategy}.json").write_text(json.dumps({
        "best_value": study.best_value,
        "best_params": study.best_params,
        "n_trials": args.trials,
    }, indent=2))
    print(f"Best Sharpe: {study.best_value:.2f}")
    print(json.dumps(study.best_params, indent=2))


if __name__ == "__main__":
    main()
