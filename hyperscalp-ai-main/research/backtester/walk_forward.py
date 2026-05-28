"""
Walk-forward harness. Splits the dataset into rolling (train, test) windows;
optimizes parameters on train via grid or Optuna; evaluates on test;
concatenates the out-of-sample test trades. The OOS trades are what matter.

A strategy whose in-sample performance survives walk-forward at >80% of its
in-sample Sharpe is a candidate for live. Anything else is curve-fit.
"""
import argparse, json
from pathlib import Path
import pandas as pd
import numpy as np

from .data_loader import load_all
from .engine import BacktestConfig, simulate
from .cost_model import CostConfig
from .strategies import range_mr_signal, trend_breakout_signal, liquidation_fade_signal
from ..eval.metrics import summary as metrics_summary


REGISTRY = {
    "range_mr": (range_mr_signal, {
        "min_choppiness": [55, 60, 65], "max_adx_15m": [20, 22, 25],
        "vwap_sigma_entry": [1.8, 2.0, 2.2], "vwap_sigma_stop": [2.5, 3.0, 3.5],
        "max_funding_skew_bps": [20], "tp_at_vwap": [True], "use_post_only": [True],
    }),
    "trend_breakout": (trend_breakout_signal, {
        "min_adx_15m": [22, 25, 28], "ema_stack_required": [True],
        "breakout_lookback_min": [4, 5, 6], "require_cvd_new_extreme": [True],
        "partial_tp_r": [1.0], "partial_tp_fraction": [0.5], "chandelier_atr_mult": [2.0, 2.5, 3.0],
    }),
    "liquidation_fade": (liquidation_fade_signal, {
        "min_liquidation_usd_5s": [150_000, 250_000, 400_000], "sweep_lookback_min": [4, 5, 6],
        "require_aggressor_flip": [True], "rr_target": [1.3, 1.5, 1.8], "max_holding_minutes": [6, 8, 10],
    }),
}


def grid(params: dict):
    from itertools import product
    keys = list(params.keys()); vals = [params[k] for k in keys]
    for combo in product(*vals):
        yield dict(zip(keys, combo))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--strategy", required=True, choices=list(REGISTRY.keys()))
    p.add_argument("--data", required=True)
    p.add_argument("--coins", default="BTC,ETH,SOL")
    p.add_argument("--n-splits", type=int, default=4)
    p.add_argument("--train-frac", type=float, default=0.6)
    p.add_argument("--out", default="reports/walkforward")
    args = p.parse_args()

    coins = [c.strip() for c in args.coins.split(",")]
    data = load_all(Path(args.data), coins)
    fn, search_space = REGISTRY[args.strategy]
    bt_cfg = BacktestConfig(cost=CostConfig())

    # Determine the global time range
    all_bars = pd.concat([data[c]["bars"] for c in coins if not data[c]["bars"].empty])
    if all_bars.empty:
        print("No data."); return
    all_bars = all_bars.sort_index()
    t_start = all_bars.index.min(); t_end = all_bars.index.max()
    span = (t_end - t_start) / args.n_splits

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    all_oos = []
    splits_summary = []
    for i in range(args.n_splits):
        s = t_start + span * i; e = s + span
        train_end = s + (e - s) * args.train_frac
        # Optimize on [s, train_end]
        best = None; best_sharpe = -1e9; best_cfg = None
        for params in grid(search_space):
            in_trades = []
            for coin in coins:
                bars = data[coin]["bars"]
                trades = data[coin]["trades"]
                if bars.empty:
                    continue
                window = {c: dict(data[c]) for c in coins}
                window[coin]["bars"] = bars.loc[s:train_end]
                window[coin]["trades"] = trades.loc[(trades["ts"] >= s) & (trades["ts"] <= train_end)] if not trades.empty else trades
                df = simulate(window, coin, fn, {"cfg": params}, bt_cfg)
                if not df.empty:
                    in_trades.append(df)
            if not in_trades:
                continue
            full = pd.concat(in_trades, ignore_index=True)
            sharpe = float(full["r_multiple"].mean() / full["r_multiple"].std() * np.sqrt(252 * 10)) if full["r_multiple"].std() > 0 else 0
            if sharpe > best_sharpe and len(full) >= 10:
                best_sharpe = sharpe; best = full; best_cfg = params
        if best_cfg is None:
            print(f"[split {i}] no in-sample winners"); continue
        # OOS [train_end, e]
        oos_trades = []
        for coin in coins:
            bars = data[coin]["bars"]; trades = data[coin]["trades"]
            if bars.empty: continue
            window = {c: dict(data[c]) for c in coins}
            window[coin]["bars"] = bars.loc[train_end:e]
            window[coin]["trades"] = trades.loc[(trades["ts"] >= train_end) & (trades["ts"] <= e)] if not trades.empty else trades
            df = simulate(window, coin, fn, {"cfg": best_cfg}, bt_cfg)
            if not df.empty:
                oos_trades.append(df)
        if oos_trades:
            full = pd.concat(oos_trades, ignore_index=True)
            full["split"] = i
            all_oos.append(full)
            print(f"[split {i}] in_sharpe={best_sharpe:.2f} oos_n={len(full)} oos_pnl=${full['pnl_usd'].sum():.2f} cfg={best_cfg}")
            splits_summary.append({"split": i, "in_sharpe": best_sharpe, "oos_summary": metrics_summary(full), "cfg": best_cfg})

    if all_oos:
        full = pd.concat(all_oos, ignore_index=True)
        full.to_parquet(out / f"oos_{args.strategy}.parquet", index=False)
        summary = metrics_summary(full)
        summary["splits"] = splits_summary
        (out / f"summary_{args.strategy}.json").write_text(json.dumps(summary, indent=2, default=str))
        print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
