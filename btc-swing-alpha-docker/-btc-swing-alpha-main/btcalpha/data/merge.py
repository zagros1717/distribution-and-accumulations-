"""
btcalpha.data.merge
~~~~~~~~~~~~~~~~~~~
ادغام داده‌ی چند صرافی به یک سری‌ی «اجماعی» (consensus) و چسباندن ماکرو.

Leakage rule:
  این ماژول هرگز از backfill استفاده نمی‌کند. در تایم‌سریز مالی، bfill یعنی
  آوردن مقدار آینده به گذشته و باعث look-ahead leakage می‌شود. فقط ffill مجاز
  است و ردیف‌هایی که OHLCV معتبر ندارند حذف می‌شوند.
"""
from __future__ import annotations

from typing import Dict

import pandas as pd

from btcalpha.config import get_logger
from btcalpha.data.exchanges import fetch_all_exchanges
from btcalpha.data.macro import align_macro_to, fetch_macro

log = get_logger("data.merge")


def _consensus_ohlcv(by_exchange: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    از چند DataFrame صرافی، یک سری‌ی اجماعی می‌سازد.
    افزون بر OHLCV، دو فیچر بین‌صرافی اضافه می‌کند:
      - x_spread: پراکندگی نسبی قیمت بین صرافی‌ها (٪)
      - x_count : تعداد صرافی‌هایی که در آن کندل داده داشتند
    """
    all_index = sorted(set().union(*[df.index for df in by_exchange.values()]))
    all_index = pd.DatetimeIndex(all_index)

    closes = pd.DataFrame(index=all_index)
    opens = pd.DataFrame(index=all_index)
    highs = pd.DataFrame(index=all_index)
    lows = pd.DataFrame(index=all_index)
    vols = pd.DataFrame(index=all_index)

    for ex, df in by_exchange.items():
        d = df.reindex(all_index)
        closes[ex] = d["close"]
        opens[ex] = d["open"]
        highs[ex] = d["high"]
        lows[ex] = d["low"]
        vols[ex] = d["volume"]

    consensus = pd.DataFrame(index=all_index)
    consensus["open"] = opens.median(axis=1)
    consensus["high"] = highs.median(axis=1)
    consensus["low"] = lows.median(axis=1)
    consensus["close"] = closes.median(axis=1)
    consensus["volume"] = vols.sum(axis=1, min_count=1)

    consensus["x_spread"] = (
        (closes.max(axis=1) - closes.min(axis=1)) / consensus["close"] * 100
    )
    consensus["x_count"] = closes.notna().sum(axis=1)

    consensus = consensus.dropna(subset=["open", "high", "low", "close"])
    consensus = consensus.ffill()
    return consensus


def build_dataset(timeframe: str) -> pd.DataFrame:
    """
    دیتاست کامل و آماده‌ی یک تایم‌فریم را می‌سازد:
      OHLCV اجماعی + فیچرهای بین‌صرافی + ستون‌های ماکرو.

    خروجی این تابع causal است: هیچ مقدار آینده‌ای به گذشته backfill نمی‌شود.
    """
    log.info("ساخت دیتاست برای تایم‌فریم %s ...", timeframe)
    by_exchange = fetch_all_exchanges(timeframe)
    log.info("صرافی‌های دریافت‌شده: %s", list(by_exchange.keys()))

    consensus = _consensus_ohlcv(by_exchange)
    log.info(
        "سری‌ی اجماعی: %d کندل (%s تا %s)",
        len(consensus), consensus.index[0], consensus.index[-1],
    )

    macro = fetch_macro()
    if not macro.empty:
        macro_aligned = align_macro_to(consensus.index, macro).add_prefix("macro_")
        dataset = consensus.join(macro_aligned)
        log.info("ماکرو چسبانده شد: %s", list(macro.columns))
    else:
        dataset = consensus
        log.warning("بدون ماکرو ادامه می‌دهیم.")

    # فقط forward-fill: bfill در تایم‌سریز look-ahead leakage ایجاد می‌کند.
    dataset = dataset.ffill()
    dataset = dataset.dropna(subset=["open", "high", "low", "close", "volume"])
    dataset.attrs["timeframe"] = timeframe
    dataset.attrs["leakage_safe_fill"] = "ffill_only_no_bfill"
    return dataset
