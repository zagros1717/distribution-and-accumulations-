"""
btcalpha.data.macro
~~~~~~~~~~~~~~~~~~~
دریافت داده‌ی ماکرو بازاری از yfinance:
  DXY (شاخص دلار)، VIX (ترس بازار)، طلا، بازده اوراق ۱۰ساله، S&P 500.

این داده‌ها برای تشخیص رژیم اقتصادی (رکود/رونق) و به‌عنوان فیچر
به مدل داده می‌شوند. همگی روزانه‌اند و در صورت نیاز به تایم‌فریم
دیگر forward-fill می‌شوند.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd

from btcalpha.config import (
    get_config, get_logger, resolve_path, cache_read, cache_write, cache_exists
)

log = get_logger("data.macro")


def _fetch_yf(ticker: str, days: int) -> Optional[pd.DataFrame]:
    """یک نماد را از yfinance می‌گیرد. در صورت خطا None."""
    try:
        import yfinance as yf  # noqa
    except ImportError:
        log.warning("کتابخانه‌ی yfinance نصب نیست. `pip install yfinance`")
        return None

    import yfinance as yf

    start = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
    try:
        df = yf.download(
            ticker, start=start, progress=False, auto_adjust=True, threads=False
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("خطا در دریافت %s از yfinance: %s", ticker, exc)
        return None

    if df is None or df.empty:
        return None

    # yfinance گاهی MultiIndex برمی‌گرداند
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Close"]].rename(columns={"Close": "close"})
    df.index = pd.to_datetime(df.index, utc=True)
    df.index.name = "timestamp"
    return df


def _synthetic_macro(name: str, days: int, seed: int) -> pd.DataFrame:
    """داده‌ی ماکرو شبیه‌سازی‌شده — فقط برای تست ساختار."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(
        end=datetime.now(timezone.utc).date(), periods=days, freq="D", tz="UTC"
    )
    # مقادیر پایه‌ی واقع‌نما برای هر شاخص
    base = {"DXY": 103, "VIX": 18, "GOLD": 2000, "US10Y": 4.2, "SPX": 4500}.get(
        name, 100
    )
    vol = {"DXY": 0.004, "VIX": 0.06, "GOLD": 0.009, "US10Y": 0.02, "SPX": 0.011}.get(
        name, 0.01
    )
    log_ret = rng.normal(0, vol, days)
    series = base * np.exp(np.cumsum(log_ret))
    df = pd.DataFrame({"close": series}, index=idx)
    df.index.name = "timestamp"
    return df


def fetch_macro() -> pd.DataFrame:
    """
    همه‌ی شاخص‌های ماکرو را می‌گیرد و در یک DataFrame واحد ادغام می‌کند.
    خروجی: ستون‌ها به نام شاخص‌ها (DXY, VIX, GOLD, US10Y, SPX)، ایندکس روزانه.
    """
    cfg = get_config()
    macro_cfg = cfg["data"]["macro"]

    if not macro_cfg.get("enabled", True):
        log.info("ماکرو غیرفعال است.")
        return pd.DataFrame()

    cache_dir = resolve_path(cfg["data"]["cache_dir"])
    cache_base = cache_dir / "macro"

    # کش ماکرو ۱۲ ساعت معتبر است
    existing = cache_exists(cache_base)
    if existing is not None:
        import time

        age_h = (time.time() - existing.stat().st_mtime) / 3600
        if age_h < 12:
            log.info("بارگذاری ماکرو از کش.")
            return cache_read(cache_base)

    days = macro_cfg["history_days"]
    frames = {}
    for name, ticker in macro_cfg["tickers"].items():
        log.info("دریافت ماکرو: %s (%s)", name, ticker)
        df = _fetch_yf(ticker, days)
        if df is None or df.empty:
            if cfg["data"].get("allow_synthetic_fallback", False):
                log.warning("ماکرو %s ناموفق — داده‌ی شبیه‌سازی‌شده.", name)
                df = _synthetic_macro(name, days, seed=7 + hash(name) % 100)
            else:
                log.error("ماکرو %s نادیده گرفته شد.", name)
                continue
        frames[name] = df["close"].rename(name)

    if not frames:
        log.warning("هیچ داده‌ی ماکرویی به‌دست نیامد.")
        return pd.DataFrame()

    macro = pd.concat(frames.values(), axis=1)
    # روزهای بازارهای سنتی تعطیل‌اند؛ forward-fill می‌کنیم
    macro = macro.sort_index().ffill().dropna(how="all")
    cache_write(macro, cache_base)
    return macro


def align_macro_to(index: pd.DatetimeIndex, macro: pd.DataFrame) -> pd.DataFrame:
    """
    داده‌ی ماکرو روزانه را به ایندکس دلخواه (مثلاً ۴ساعته‌ی BTC) هم‌تراز می‌کند.
    از forward-fill استفاده می‌شود: هر کندل، آخرین مقدار ماکرویِ در دسترس را می‌گیرد.
    این کار از «نشت داده از آینده» جلوگیری می‌کند.
    """
    if macro.empty:
        return pd.DataFrame(index=index)
    aligned = macro.reindex(macro.index.union(index)).ffill().reindex(index)
    return aligned
