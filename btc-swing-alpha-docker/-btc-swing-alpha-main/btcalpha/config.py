"""
btcalpha.config
~~~~~~~~~~~~~~~
بارگذاری پیکربندی مرکزی + ابزارهای مشترک (لاگ، مسیرها).
همه‌ی ماژول‌ها از این‌جا `get_config()` را صدا می‌زنند.
"""
from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import yaml

# ریشه‌ی پروژه = دو پوشه بالاتر از این فایل
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


@lru_cache(maxsize=1)
def get_config() -> Dict[str, Any]:
    """پیکربندی را یک‌بار می‌خواند و کش می‌کند."""
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"فایل پیکربندی پیدا نشد: {CONFIG_PATH}")
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # مسیرهای نسبی را مطلق کن
    cfg["_project_root"] = str(PROJECT_ROOT)
    return cfg


def resolve_path(relative: str) -> Path:
    """مسیر نسبی داخل config را به مسیر مطلق تبدیل می‌کند و می‌سازد."""
    p = PROJECT_ROOT / relative
    p.mkdir(parents=True, exist_ok=True)
    return p


def cache_read(path: Path) -> "Any":
    """خواندن کش — parquet اگر در دسترس باشد، وگرنه pickle."""
    import pandas as pd

    pq = path.with_suffix(".parquet")
    pk = path.with_suffix(".pkl")
    if pq.exists():
        try:
            return pd.read_parquet(pq)
        except Exception:  # noqa: BLE001
            pass
    if pk.exists():
        return pd.read_pickle(pk)
    raise FileNotFoundError(path)


def cache_write(df: "Any", path: Path) -> None:
    """نوشتن کش — تلاش برای parquet، در صورت نبود pyarrow از pickle."""
    pq = path.with_suffix(".parquet")
    pk = path.with_suffix(".pkl")
    try:
        df.to_parquet(pq)
    except Exception:  # noqa: BLE001  (pyarrow/fastparquet نصب نیست)
        df.to_pickle(pk)


def cache_exists(path: Path):
    """آیا کش (به هر فرمتی) وجود دارد؟ مسیر موجود را برمی‌گرداند یا None."""
    for ext in (".parquet", ".pkl"):
        p = path.with_suffix(ext)
        if p.exists():
            return p
    return None


@lru_cache(maxsize=8)
def get_logger(name: str = "btcalpha") -> logging.Logger:
    """لاگر مشترک با خروجی هم‌زمان روی کنسول و فایل."""
    cfg = get_config()
    logger = logging.getLogger(name)
    if logger.handlers:  # قبلاً ساخته شده
        return logger

    level = getattr(logging, cfg["logging"]["level"].upper(), logging.INFO)
    logger.setLevel(level)

    fmt = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # کنسول
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # فایل
    log_dir = resolve_path(cfg["logging"]["dir"])
    fh = logging.FileHandler(log_dir / "btcalpha.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger
