"""
btcalpha.data.exchanges
~~~~~~~~~~~~~~~~~~~~~~~
Real OHLCV fetcher with persistent/incremental cache.
"""
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from btcalpha.config import (
    cache_exists,
    cache_read,
    cache_write,
    get_config,
    get_logger,
    resolve_path,
)

log = get_logger("data.exchanges")

_SYMBOL_MAP = {
    "bitfinex": "BTC/USD",
    "kraken": "BTC/USD",
    "coinbase": "BTC/USD",
    "binance": "BTC/USDT",
}

_TF_MS = {
    "1m": 60_000,
    "5m": 300_000,
    "15m": 900_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
}

_BITFINEX_TF_MAP = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "1h": "1h",
    "4h": "1h",
    "1d": "1D",
}

_CACHE_MAX_AGE_SECONDS = {
    "5m": 4 * 60,
    "15m": 12 * 60,
    "1h": 45 * 60,
    "4h": 3 * 3600,
    "1d": 12 * 3600,
}


def _parse_timestamp_ms(raw: str | None) -> int | None:
    if not raw:
        return None
    try:
        ts = pd.Timestamp(raw)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        return int(ts.timestamp() * 1000)
    except Exception as exc:  # noqa: BLE001
        log.warning("history_start نامعتبر است: %s (%s)", raw, exc)
        return None


def _parse_history_start(cfg: dict, timeframe: str) -> int | None:
    data_cfg = cfg.get("data", {})
    by_tf = data_cfg.get("history_start_by_tf", {}) or {}
    raw = by_tf.get(timeframe) or data_cfg.get("history_start")
    return _parse_timestamp_ms(raw)


def _make_exchange(name: str):
    try:
        import ccxt  # noqa
    except ImportError:
        log.warning("کتابخانه‌ی ccxt نصب نیست. `pip install ccxt`")
        return None

    import ccxt
    klass = getattr(ccxt, name, None)
    if klass is None:
        log.error("صرافی پشتیبانی نمی‌شود: %s", name)
        return None
    return klass({"enableRateLimit": True, "timeout": 30_000})


def _rows_to_df(rows: List[list]) -> Optional[pd.DataFrame]:
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df.drop_duplicates("timestamp").set_index("timestamp").sort_index()


def _resample_ohlcv(df: pd.DataFrame, timeframe: str, limit: int) -> pd.DataFrame:
    if timeframe != "4h":
        return df.tail(limit)
    out = df.resample("4h", origin="start_day").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna(subset=["open", "high", "low", "close"])
    return out.tail(limit)


def _bitfinex_rest_url(tf: str, start_ms: int, end_ms: int, limit: int) -> str:
    params = urllib.parse.urlencode({
        "start": int(start_ms),
        "end": int(end_ms),
        "limit": int(limit),
        "sort": 1,
    })
    return f"https://api-pub.bitfinex.com/v2/candles/trade:{tf}:tBTCUSD/hist?{params}"


def _fetch_json_url(url: str, timeout: int = 30):
    req = urllib.request.Request(url, headers={"User-Agent": "btc-swing-alpha/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - public market data endpoint
        return json.loads(resp.read().decode("utf-8"))


def fetch_latest_price(exchange_name: str = "bitfinex") -> dict:
    """Fetch a live spot price for display and live-entry reference.

    This is intentionally separate from OHLCV candle data. Daily candles may show
    the latest 1d candle close, which is not always the same as the current market
    price shown on exchanges.
    """
    ts = datetime.now(timezone.utc).isoformat()

    if exchange_name == "bitfinex":
        try:
            raw = _fetch_json_url("https://api-pub.bitfinex.com/v2/ticker/tBTCUSD", timeout=10)
            # Bitfinex v2 ticker row:
            # [BID, BID_SIZE, ASK, ASK_SIZE, DAILY_CHANGE, DAILY_CHANGE_RELATIVE,
            #  LAST_PRICE, VOLUME, HIGH, LOW]
            return {
                "exchange": "bitfinex",
                "symbol": "BTC/USD",
                "price": float(raw[6]),
                "bid": float(raw[0]),
                "ask": float(raw[2]),
                "spread": float(raw[2]) - float(raw[0]),
                "timestamp": ts,
                "source": "ticker",
            }
        except Exception as exc:  # noqa: BLE001
            log.warning("Bitfinex ticker failed: %s", exc)

    ex = _make_exchange(exchange_name)
    if ex is not None:
        try:
            ticker = ex.fetch_ticker(_SYMBOL_MAP.get(exchange_name, "BTC/USD"))
            return {
                "exchange": exchange_name,
                "symbol": _SYMBOL_MAP.get(exchange_name, "BTC/USD"),
                "price": float(ticker.get("last") or ticker.get("close")),
                "bid": float(ticker["bid"]) if ticker.get("bid") is not None else None,
                "ask": float(ticker["ask"]) if ticker.get("ask") is not None else None,
                "spread": float(ticker["ask"] - ticker["bid"]) if ticker.get("ask") is not None and ticker.get("bid") is not None else None,
                "timestamp": ts,
                "source": "ticker",
            }
        except Exception as exc:  # noqa: BLE001
            log.warning("ticker failed for %s: %s", exchange_name, exc)

    return {"exchange": exchange_name, "symbol": _SYMBOL_MAP.get(exchange_name, "BTC/USD"), "price": None, "timestamp": ts, "source": "unavailable"}


def _expected_rows_since(since_ms: int | None, timeframe: str) -> int | None:
    if since_ms is None:
        return None
    now_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
    if now_ms <= since_ms:
        return 0
    return int((now_ms - since_ms) / _TF_MS[timeframe]) + 1


def _fetch_bitfinex_rest(timeframe: str, limit: int, since_ms: int | None = None) -> Optional[pd.DataFrame]:
    if timeframe not in _BITFINEX_TF_MAP:
        return None

    source_tf = _BITFINEX_TF_MAP[timeframe]
    source_tf_ms = _TF_MS["1h"] if timeframe == "4h" else _TF_MS[timeframe]
    target_limit = int(limit)
    source_limit = target_limit * 4 if timeframe == "4h" else target_limit
    now_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
    since = int(since_ms) if since_ms is not None else now_ms - source_limit * source_tf_ms

    rows: list[list] = []
    page = 0
    max_pages = max(5, int(np.ceil(source_limit / 10_000)) + 5)

    log.info(
        "Bitfinex REST backfill %s via %s | target=%d source_limit=%d since=%s",
        timeframe,
        source_tf,
        target_limit,
        source_limit,
        pd.to_datetime(since, unit="ms", utc=True),
    )

    while since < now_ms and len(rows) < source_limit and page < max_pages:
        batch_limit = min(10_000, source_limit - len(rows))
        url = _bitfinex_rest_url(source_tf, since, now_ms, batch_limit)
        try:
            raw = _fetch_json_url(url)
        except Exception as exc:  # noqa: BLE001
            log.warning("Bitfinex REST error page=%d tf=%s: %s", page, timeframe, exc)
            break

        if not raw:
            break

        converted = []
        for r in raw:
            if len(r) < 6:
                continue
            converted.append([r[0], r[1], r[3], r[4], r[2], r[5]])

        if not converted:
            break

        rows.extend(converted)
        last_ts = int(converted[-1][0])
        next_since = last_ts + source_tf_ms
        if next_since <= since:
            break
        since = next_since
        page += 1
        if page % 10 == 0:
            log.info("Bitfinex %s progress: rows=%d page=%d last=%s", timeframe, len(rows), page, pd.to_datetime(last_ts, unit="ms", utc=True))
        time.sleep(0.25)

    df = _rows_to_df(rows)
    if df is None or df.empty:
        return None
    return _resample_ohlcv(df, timeframe, target_limit)


def _fetch_ccxt(exchange_name: str, timeframe: str, limit: int, since_ms: int | None = None) -> Optional[pd.DataFrame]:
    if exchange_name == "bitfinex":
        return _fetch_bitfinex_rest(timeframe, limit, since_ms=since_ms)

    ex = _make_exchange(exchange_name)
    if ex is None:
        return None

    symbol = _SYMBOL_MAP.get(exchange_name, "BTC/USD")
    tf_ms = _TF_MS[timeframe]
    all_rows: List[list] = []
    now_ms = ex.milliseconds()
    since = since_ms if since_ms is not None else now_ms - limit * tf_ms
    hard_limit = int(limit)

    try:
        while since < now_ms and len(all_rows) < hard_limit:
            remaining = hard_limit - len(all_rows)
            batch_limit = min(1000, remaining)
            batch = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=batch_limit)
            if not batch:
                break
            all_rows.extend(batch)
            next_since = batch[-1][0] + tf_ms
            if next_since <= since:
                break
            since = next_since
            time.sleep(ex.rateLimit / 1000.0)
    except Exception as exc:  # noqa: BLE001
        log.warning("خطا در دریافت از %s: %s", exchange_name, exc)
        return None

    df = _rows_to_df(all_rows)
    if df is None or df.empty:
        return None
    return df.tail(limit)


def _append_cached_with_new(existing_df: pd.DataFrame, new_df: Optional[pd.DataFrame], limit: int) -> pd.DataFrame:
    if new_df is None or new_df.empty:
        return existing_df.tail(limit)
    merged = pd.concat([existing_df, new_df]).sort_index()
    merged = merged[~merged.index.duplicated(keep="last")]
    return merged.tail(limit)


def _cache_has_enough_rows(existing_df: pd.DataFrame, limit: int, timeframe: str, since_ms: int | None = None) -> bool:
    expected = _expected_rows_since(since_ms, timeframe)
    if expected is not None:
        required = max(500, int(min(limit, expected) * 0.85))
    else:
        required = max(500, int(limit * 0.90))
    return len(existing_df) >= required


def _cache_starts_near_target(existing_df: pd.DataFrame, since_ms: int | None, timeframe: str) -> bool:
    if since_ms is None or existing_df is None or existing_df.empty:
        return True
    first_ms = int(existing_df.index[0].timestamp() * 1000)
    tolerance = 5 * _TF_MS[timeframe]
    return first_ms <= since_ms + tolerance


def fetch_ohlcv(
    exchange_name: str,
    timeframe: str,
    limit: int,
    use_cache: bool = True,
    since_ms: int | None = None,
    require_full_history: bool = True,
) -> pd.DataFrame:
    cfg = get_config()
    cache_dir = resolve_path(cfg["data"]["cache_dir"])
    cache_base = cache_dir / f"{exchange_name}_{timeframe}"
    existing_path = cache_exists(cache_base) if use_cache else None
    existing_df = None

    if existing_path is not None:
        existing_df = cache_read(cache_base)
        age_s = time.time() - existing_path.stat().st_mtime
        max_age_s = _CACHE_MAX_AGE_SECONDS.get(timeframe, 30 * 60)
        enough_rows = _cache_has_enough_rows(existing_df, limit, timeframe, since_ms) if require_full_history else len(existing_df) >= min(limit, 500)
        starts_ok = _cache_starts_near_target(existing_df, since_ms, timeframe) if require_full_history else True

        if enough_rows and starts_ok and age_s < max_age_s:
            log.info("بارگذاری از کش تازه و کامل: %s | rows=%d | age=%.1fs", existing_path.name, len(existing_df), age_s)
            return existing_df.tail(limit)

        if not enough_rows or not starts_ok:
            log.warning(
                "کش ناقص است: %s | rows=%d requested=%d starts_ok=%s — backfill کامل انجام می‌شود",
                existing_path.name,
                len(existing_df),
                limit,
                starts_ok,
            )
            existing_df = None

    if existing_df is not None and not existing_df.empty:
        last_ts = existing_df.index[-1]
        fetch_since = int(last_ts.timestamp() * 1000) + _TF_MS[timeframe]
        now_ms = int(time.time() * 1000)
        missing_estimate = max(50, min(10_000, int((now_ms - fetch_since) / _TF_MS[timeframe]) + 20))
        log.info("آپدیت incremental %s | %s | از %s | حدود %d کندل", exchange_name, timeframe, last_ts, missing_estimate)
        new_df = _fetch_ccxt(exchange_name, timeframe, missing_estimate, since_ms=fetch_since)
        df = _append_cached_with_new(existing_df, new_df, limit)
    else:
        fetch_since = since_ms
        if fetch_since is None:
            fetch_since = int(time.time() * 1000) - int(limit) * _TF_MS[timeframe]
        log.info("دریافت کامل %s | %s | %d کندل از %s ...", exchange_name, timeframe, limit, pd.to_datetime(fetch_since, unit="ms", utc=True))
        df = _fetch_ccxt(exchange_name, timeframe, limit, since_ms=fetch_since)

    if df is None or df.empty:
        if existing_path is not None:
            fallback_df = cache_read(cache_base)
            log.warning("داده جدید نیامد؛ استفاده از کش قبلی %s | rows=%d", timeframe, len(fallback_df))
            df = fallback_df.tail(limit)
        else:
            raise ConnectionError(f"دریافت داده از {exchange_name} ناموفق بود.")

    if use_cache:
        cache_write(df.tail(limit), cache_base)
    return df.tail(limit)


def fetch_all_exchanges(timeframe: str) -> Dict[str, pd.DataFrame]:
    cfg = get_config()
    data_cfg = cfg["data"]
    primary = data_cfg.get("primary_exchange")
    full_limit = int(data_cfg["history"][timeframe])
    backup_limit = int(data_cfg.get("backup_history_limit", 1500))
    since_ms = _parse_history_start(cfg, timeframe)

    out: Dict[str, pd.DataFrame] = {}
    for ex in data_cfg["exchanges"]:
        try:
            is_primary = ex == primary
            ex_limit = full_limit if is_primary else min(full_limit, backup_limit)
            out[ex] = fetch_ohlcv(
                ex,
                timeframe,
                ex_limit,
                since_ms=since_ms if is_primary else None,
                require_full_history=is_primary,
            )
            log.info("%s %s دریافت شد | rows=%d | first=%s | last=%s", ex, timeframe, len(out[ex]), out[ex].index[0], out[ex].index[-1])
        except Exception as exc:  # noqa: BLE001
            log.error("صرافی %s نادیده گرفته شد: %s", ex, exc)
    if not out:
        raise RuntimeError("هیچ صرافی‌ای داده نداد.")
    return out
