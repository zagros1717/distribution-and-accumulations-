"""
Safety enforcement.

This module is imported at the top of every entrypoint. If anyone tries to
flip `safety.execution_enabled` to true, or tries to wire in a private API
client, the program refuses to start. There is no command-line override.

Trading capability is OUT OF SCOPE for this project. This is research only.
"""
from __future__ import annotations

from typing import Mapping


class ExecutionDisabledError(RuntimeError):
    """Raised when something in the system pretends it can place orders."""


def assert_research_mode(config: Mapping) -> None:
    """Call once at program start. Raises if the config has been tampered with."""
    safety = config.get("safety", {})
    if safety.get("execution_enabled", False):
        raise ExecutionDisabledError(
            "execution_enabled=true is not permitted in this project. "
            "This system is research-only by design."
        )
    if safety.get("allow_private_api", False):
        raise ExecutionDisabledError(
            "allow_private_api=true is not permitted. No private API keys."
        )
    if safety.get("allow_withdrawals", False):
        raise ExecutionDisabledError(
            "allow_withdrawals=true is not permitted. This code never moves funds."
        )


def forbid_trading_call(what: str) -> None:
    """
    Defensive guard. Any code path that *looks* like it places orders should
    call this. It always raises. The function exists so static analyzers and
    grep can find these landmines quickly.
    """
    raise ExecutionDisabledError(
        f"Refusing to perform trading-related action: {what!r}. "
        "This project has no execution layer."
    )
