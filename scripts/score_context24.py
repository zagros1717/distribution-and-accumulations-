#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.context24.calibration import load_calibration_csv
from src.context24.report import signal_to_dict, write_signal_report
from src.context24.scoring import score_table


def main() -> int:
    parser = argparse.ArgumentParser(description="Score a 24h market-context table with quality and calibration gates.")
    parser.add_argument("--input", required=True, help="CSV file with source, metric, value, as_of, fetched_at columns")
    parser.add_argument("--history", default=None, help="Optional CSV with metric_key,value,forward_return_24h")
    parser.add_argument("--out", default="data/reports/context24.md", help="Markdown report output path")
    parser.add_argument("--min-samples", type=int, default=50)
    args = parser.parse_args()

    rows = pd.read_csv(args.input).to_dict(orient="records")
    calibration = load_calibration_csv(args.history, min_samples=args.min_samples) if args.history else {}
    signal = score_table(rows, calibration=calibration)
    write_signal_report(signal, args.out)
    print(json.dumps(signal_to_dict(signal), indent=2, default=str))
    return 0 if signal.status.startswith("CONFIRMED") else 1


if __name__ == "__main__":
    raise SystemExit(main())
