"""pytest conftest: ensure the repo root is importable.

We rely on `from src.xxx import yyy` everywhere. Pytest discovers tests from a
working directory, but it doesn't automatically put the project root on
sys.path unless we ask. This conftest does exactly that.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
