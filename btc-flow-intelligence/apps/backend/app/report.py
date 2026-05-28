"""Markdown report generator (the 9-section structure from the brief)."""

from __future__ import annotations

import datetime as dt

from app.schemas import SignalReading
from app.scoring import ScoringResult

_VERDICT_EMOJI = {"Accumulation": "🟢", "Distribution": "🔴", "Mixed/Neutral": "🟡"}


def _fmt(v: float | None, nd: int = 2) -> str:
    return "—" if v is None else f"{v:,.{nd}f}"


def _evidence(signals: list[SignalReading], positive: bool) -> list[SignalReading]:
    return [s for s in signals if (s.score > 0 if positive else s.score < 0)]


def generate_report(
    *,
    result: ScoringResult,
    signals: list[SignalReading],
    btc_price: float | None,
    btc_change_24h: float | None,
    history: list[dict] | None = None,
) -> str:
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    emoji = _VERDICT_EMOJI.get(result.verdict, "")
    lines: list[str] = []

    # 1. Executive summary
    lines += [
        f"# BTC Flow Intelligence — 24h Report",
        f"_Generated {now}_",
        "",
        "## 1. Executive Summary",
        f"- **Verdict:** {emoji} **{result.verdict}**  ·  **Score:** `{result.final_score:+.2f}`  ·  **Confidence:** {result.confidence}",
        f"- **BTC price:** ${_fmt(btc_price, 0)} ({_fmt(btc_change_24h)}% 24h)",
        f"- **Data quality:** {result.data_quality*100:.0f}% of weighted categories backed by live sources.",
        "",
    ]

    # 2. Confirmed 24h data table
    lines += ["## 2. Confirmed 24h Data", "", "| Source | Metric | Value | Δ24h | Score | Mode |", "|---|---|---:|---:|:---:|:---:|"]
    for s in sorted(signals, key=lambda x: (x.category, x.source)):
        mode = "live" if s.is_live else "mock"
        lines.append(
            f"| {s.source} | {s.metric} | {_fmt(s.value)} | {_fmt(s.change_24h)} | {s.score:+d} | {mode} |"
        )
    lines.append("")

    # 3. Signal matrix
    lines += ["## 3. Signal Matrix", "", "| Category | Weight | Score | Weighted | Live |", "|---|---:|---:|---:|:---:|"]
    for c in result.categories:
        lines.append(
            f"| {c.label} | {c.weight*100:.0f}% | {c.score:+.2f} | {c.weighted:+.3f} | {'✓' if c.is_live else '—'} |"
        )
    lines += [f"| **Final** | **100%** | | **{result.final_score:+.2f}** | |", ""]

    # 4 / 5. Accumulation & distribution evidence
    acc = _evidence(signals, True)
    dist = _evidence(signals, False)
    lines += ["## 4. Accumulation Evidence"]
    lines += ([f"- {s.source} · {s.metric}: {_fmt(s.value)} (score {s.score:+d})" for s in acc] or ["- None significant."])
    lines += ["", "## 5. Distribution Evidence"]
    lines += ([f"- {s.source} · {s.metric}: {_fmt(s.value)} (score {s.score:+d})" for s in dist] or ["- None significant."])

    # 6. Neutral / conflicting
    neutral = [s for s in signals if s.score == 0]
    lines += ["", "## 6. Neutral / Conflicting Evidence"]
    lines += ([f"- {s.source} · {s.metric}: {_fmt(s.value)}" for s in neutral] or ["- None."])

    # 7. 7-day context
    lines += ["", "## 7. 7-Day Context"]
    if history:
        lines += ["", "| Time (UTC) | Score | Verdict |", "|---|---:|:---:|"]
        for h in history[:7]:
            ts = h["timestamp"][:16].replace("T", " ")
            lines.append(f"| {ts} | {h['final_score']:+.2f} | {h['verdict']} |")
    else:
        lines.append("- Insufficient history (first runs). Context accrues hourly.")

    # 8. Access limitations
    live_sources = sorted({s.source for s in signals if s.is_live})
    mock_sources = sorted({s.source for s in signals if not s.is_live})
    lines += [
        "",
        "## 8. Access Limitations",
        f"- **Live sources:** {', '.join(live_sources) or 'none (mock mode)'}",
        f"- **Mock / paywalled sources:** {', '.join(mock_sources) or 'none'}",
        "- Paywalled feeds (CoinGlass, CryptoQuant, Arkham, Kaiko, CME) require API keys; until configured they serve realistic mock data and are discounted from the confidence score.",
    ]

    # 9. Final verdict
    lines += [
        "",
        "## 9. Final Verdict",
        f"> {emoji} **{result.verdict}** at score `{result.final_score:+.2f}` "
        f"with **{result.confidence}** confidence.",
        "",
        "_This is automated market-structure analysis, not financial advice._",
    ]
    return "\n".join(lines)
