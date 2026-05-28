"""Optional Telegram alerts. Silently no-ops when not configured."""

from __future__ import annotations

import logging

import httpx

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


async def send_telegram(text: str) -> bool:
    if not (settings.telegram_bot_token and settings.telegram_chat_id):
        return False
    url = f"https://api.telegram.org/bot{settings.telegram_bot_token}/sendMessage"
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            await client.post(url, json={
                "chat_id": settings.telegram_chat_id,
                "text": text,
                "parse_mode": "Markdown",
            })
        return True
    except Exception as exc:  # alerts must never break the pipeline
        logger.warning("Telegram alert failed: %s", exc)
        return False


async def maybe_alert_verdict_change(previous: str | None, current: str, score: float, conf: str) -> None:
    if not settings.alert_on_verdict_change:
        return
    if previous is not None and previous == current:
        return
    emoji = {"Accumulation": "🟢", "Distribution": "🔴", "Mixed/Neutral": "🟡"}.get(current, "")
    msg = (
        f"{emoji} *BTC Flow Intelligence*\n"
        f"Verdict: *{current}*  (was {previous or 'n/a'})\n"
        f"Score: `{score:+.2f}`  ·  Confidence: {conf}"
    )
    await send_telegram(msg)
