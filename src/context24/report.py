from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from src.context24.schema import FinalSignal


def signal_to_dict(signal: FinalSignal) -> dict:
    payload = asdict(signal)
    payload["rows"] = [asdict(r) for r in signal.rows]
    return payload


def write_signal_report(signal: FinalSignal, out_path: str | Path) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# 24h market context report")
    lines.append("")
    lines.append(f"- Status: **{signal.status}**")
    lines.append(f"- Direction: **{signal.direction}**")
    lines.append(f"- Final score: **{signal.final_score:.4f}**")
    lines.append(f"- Confidence: **{signal.confidence:.3f}**")
    lines.append("")

    if signal.reasons:
        lines.append("## Rejection / watch reasons")
        lines.append("")
        for reason in signal.reasons:
            lines.append(f"- {reason}")
        lines.append("")

    lines.append("## Category scores")
    lines.append("")
    lines.append("| Category | Score | Confidence |")
    lines.append("|---|---:|---:|")
    for cat, val in sorted(signal.category_scores.items()):
        lines.append(f"| {cat} | {val:.4f} | {signal.category_confidence.get(cat, 0.0):.3f} |")
    lines.append("")

    lines.append("## Metric rows")
    lines.append("")
    lines.append("| Source | Metric | Category | Value | Delta 24h | Quality | Score | Confidence | Symbol | Usable | Reasons |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---|---|---|")
    for r in signal.rows:
        reasons = "; ".join(r.reasons)
        value = "" if r.value is None else f"{r.value:.4f}"
        delta = "" if r.delta_24h is None else f"{r.delta_24h:.4f}"
        lines.append(
            f"| {r.source} | {r.metric} | {r.category} | {value} | {delta} "
            f"| {r.data_quality:.2f} | {r.signal_score:.4f} | {r.confidence:.3f} "
            f"| {r.symbol} | {r.usable} | {reasons} |"
        )

    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(signal_to_dict(signal), indent=2, default=str))
    lines.append("```")
    out.write_text("\n".join(lines))
    return out
