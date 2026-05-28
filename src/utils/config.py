"""Config loader. Always runs the safety assertion on load."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

from src.safety import assert_research_mode


def load_config(path: str | Path = "config/config.yaml") -> Dict[str, Any]:
    """Load YAML config from disk, validate safety, return as dict."""
    p = Path(path)
    if not p.is_absolute():
        # Walk up looking for the config file relative to project root.
        # This makes load_config() work from anywhere (tests, notebooks).
        candidates = [p, Path.cwd() / p, Path(__file__).resolve().parents[2] / p]
        for c in candidates:
            if c.exists():
                p = c
                break
    with open(p, "r") as f:
        cfg = yaml.safe_load(f)
    assert_research_mode(cfg)
    return cfg
