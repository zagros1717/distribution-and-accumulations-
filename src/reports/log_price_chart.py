"""
Logarithmic price chart generator.

Reads reconstructed snapshot parquet files from the local research store and
writes a static PNG chart. The default window is the trailing year, so this is
intended for longer-horizon visual review rather than the previous 24h-only
view.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Tuple