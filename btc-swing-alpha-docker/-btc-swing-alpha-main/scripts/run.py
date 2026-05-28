#!/usr/bin/env python3
"""
scripts/run.py
~~~~~~~~~~~~~~
رابط خط فرمان برای اجرای سیستم بدون نیاز به API/فرانت.

نمونه‌ها:
    python scripts/run.py pipeline --tf 1d        # اجرای کامل + گزارش بک‌تست
    python scripts/run.py pipeline --tf 4h --retrain
    python scripts/run.py live --tf 1d            # فقط تصمیم لحظه‌ای
    python scripts/run.py all                     # هر دو تایم‌فریم
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# اجازه‌ی import از ریشه‌ی پروژه
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from btcalpha.config import get_config, get_logger
from btcalpha.live.engine import run_pipeline, live_decision

log = get_logger("cli")


def _print_live(live: dict):
    d = live["decision"]
    print("\n" + "─" * 52)
    print(f"  تصمیم لحظه‌ای  —  {live['timestamp']}")
    print("─" * 52)
    print(f"  جهت           : {d['direction'].upper()}")
    print(f"  اندازه پوزیشن : {d['position']:+.3f}")
    print(f"  سیگنال خام ML : {d['raw_alpha']:+.3f}")
    print(f"  سیگنال نهایی  : {d['final_signal']:+.3f}")
    print(f"  اطمینان       : {d['confidence']:.3f}")
    print(f"  رژیم اقتصادی  : {d['regime']}")
    if d.get("stop_loss"):
        print(f"  حد ضرر        : {d['stop_loss']:.0f}")
        print(f"  حد سود        : {d['take_profit']:.0f}")
    if "comparison" in live:
        c = live["comparison"]
        print("  ── مقایسه با بک‌تست ──")
        print(f"  نرخ برد تاریخی هم‌جهت : {c['historical_win_rate_same_direction']}")
        print(f"  نرخ برد تاریخی هم‌رژیم: {c['historical_win_rate_same_regime']}")
    if "verdict" in live:
        v = live["verdict"]
        print(f"  حکم سیستم     : [{v['level']}] {v['text']}")
    print("─" * 52 + "\n")


def cmd_pipeline(tf: str, retrain: bool):
    snap = run_pipeline(tf, force_retrain=retrain)
    print(snap.backtest.summary())
    _print_live(snap.live)
    # اهمیت فیچرها
    imp = snap.model.feature_importance()
    if imp is not None:
        print("  مهم‌ترین فیچرها:")
        for name, val in imp.head(10).items():
            print(f"    {name:<22} {val:.1f}")
    print()


def cmd_live(tf: str):
    live = live_decision(tf)
    _print_live(live)


def main():
    parser = argparse.ArgumentParser(description="BTC Swing Alpha CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("pipeline", help="اجرای کامل زنجیره + بک‌تست")
    p1.add_argument("--tf", default="1d", help="تایم‌فریم (1d یا 4h)")
    p1.add_argument("--retrain", action="store_true", help="بازآموزی اجباری مدل")

    p2 = sub.add_parser("live", help="فقط تصمیم لحظه‌ای")
    p2.add_argument("--tf", default="1d")

    sub.add_parser("all", help="اجرای کامل برای همه‌ی تایم‌فریم‌ها")

    args = parser.parse_args()
    cfg = get_config()

    if args.cmd == "pipeline":
        cmd_pipeline(args.tf, args.retrain)
    elif args.cmd == "live":
        cmd_live(args.tf)
    elif args.cmd == "all":
        for tf in cfg["data"]["timeframes"]:
            cmd_pipeline(tf, retrain=False)


if __name__ == "__main__":
    main()
