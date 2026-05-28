"""
Recorder: raw-frame hook must be wired BEFORE adapter.connect().

The subscribe handshake — and on Coinbase, the first pre-snapshot frames the
venue sends as soon as the socket is open — flow through adapter.on_raw_frame.
If the recorder installs that hook AFTER connect() returns, anything that
arrived during the handshake is lost from the raw log, which means we can't
re-normalize that window if a bug is later found in the parser.

The fix is structural: `_open_stores(adapter)` (which sets on_raw_frame)
must run before `await adapter.connect()`. This test pins that invariant
down so a future refactor can't reorder it accidentally.

We verify two things:

  1. By the time adapter.connect() runs, adapter.on_raw_frame is already
     set to a callable. If a raw frame is delivered during connect(), it
     ends up in the recorder's raw store.
  2. The static call order in Recorder._run_one is _open_stores BEFORE
     await adapter.connect(). We check the AST so the assertion survives
     even if the test scaffold is mocked differently.
"""
from __future__ import annotations

import ast
import asyncio
import inspect
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator, Iterable, List

import pytest

from src.adapters.base import MarketDataAdapter
from src.recorder import Recorder
from src.schema import BookEvent


# ---------------------------------------------------------------------------
# 1. Static check: _open_stores called before await adapter.connect()
# ---------------------------------------------------------------------------

def test_open_stores_called_before_connect_in_source():
    """
    Walk the AST of Recorder._run_one and assert that the line which calls
    self._open_stores(adapter) appears *before* the first `await
    adapter.connect()`. A pure read of the source — no runtime mocking —
    so it can't be fooled by clever test fixtures.
    """
    import textwrap
    src = textwrap.dedent(inspect.getsource(Recorder._run_one))
    tree = ast.parse(src)
    func = next(n for n in ast.walk(tree)
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                and n.name == "_run_one")

    open_stores_line = None
    connect_await_line = None
    for node in ast.walk(func):
        # _open_stores(adapter) call
        if isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Attribute) and fn.attr == "_open_stores":
                if open_stores_line is None:
                    open_stores_line = node.lineno
        # await adapter.connect()
        if isinstance(node, ast.Await):
            v = node.value
            if isinstance(v, ast.Call) and isinstance(v.func, ast.Attribute) \
                    and v.func.attr == "connect":
                if connect_await_line is None:
                    connect_await_line = v.lineno

    assert open_stores_line is not None, "self._open_stores(adapter) not found in _run_one"
    assert connect_await_line is not None, "await adapter.connect() not found in _run_one"
    assert open_stores_line < connect_await_line, (
        f"_open_stores at line {open_stores_line} must precede connect() "
        f"at line {connect_await_line} — otherwise on_raw_frame is None "
        f"during the subscribe handshake and pre-snapshot frames are lost."
    )


# ---------------------------------------------------------------------------
# 2. Runtime check: frames arriving during connect() land in the raw store
# ---------------------------------------------------------------------------

class _ConnectCapturingAdapter(MarketDataAdapter):
    """
    Adapter that, in connect(), invokes self.on_raw_frame with a known
    payload. If the hook isn't wired yet, the payload is dropped — the
    test then sees an empty raw store and fails.
    """
    name = "captest"

    def __init__(self, canonical_symbol: str = "CAP-USD") -> None:
        self.canonical_symbol = canonical_symbol
        self.symbol = canonical_symbol
        self.on_raw_frame = None
        self.connect_saw_hook_set = False
        self.handshake_payload = '{"type":"subscriptions","handshake":true}'

    async def connect(self) -> None:
        # Critical: by the time connect() runs, on_raw_frame must already
        # be a callable installed by the recorder.
        self.connect_saw_hook_set = callable(self.on_raw_frame)
        if callable(self.on_raw_frame):
            # Deliver one raw frame during the handshake window so the
            # test can also observe it via the raw store.
            self.on_raw_frame("ws", self.handshake_payload)

    async def fetch_snapshot(self) -> Iterable[BookEvent]:
        return []

    async def stream_events(self) -> AsyncIterator[BookEvent]:
        # Yield nothing; the test signals shutdown right after fetch_snapshot.
        # An infinite-wait keeps the iterator open until the recorder
        # cancels its stream task.
        stop = asyncio.Event()
        await stop.wait()
        if False:  # pragma: no cover — make this an async generator
            yield  # type: ignore[unreachable]

    def normalize_message(self, raw_message: object):
        return []

    def validate_sequence(self, message: object):
        return None

    async def close(self) -> None:
        return


def _build_recorder(tmp_path: Path) -> Recorder:
    cfg = {
        "storage": {
            "root": str(tmp_path / "data"),
            "rotate_files_every_minutes": 60,
        },
        "sources": {
            "primary": {},
            "fallback": {},
            "failover": {
                "max_reconnects_before_switch": 3,
                "reconnect_backoff_seconds": [0.01],
                "heartbeat_timeout_seconds": 60.0,  # long; we don't want watchdog
            },
        },
    }
    return Recorder(cfg)


@pytest.mark.asyncio
async def test_hook_is_callable_during_adapter_connect(tmp_path):
    """
    Runtime check: by the time adapter.connect() runs, the recorder has
    already set adapter.on_raw_frame to a callable.
    """
    rec = _build_recorder(tmp_path)
    adapter = _ConnectCapturingAdapter()

    # Shut down as soon as we know the recorder finished connect+snapshot.
    async def stopper():
        # Wait a moment for connect() to have run.
        for _ in range(100):
            if adapter.connect_saw_hook_set:
                rec.request_shutdown()
                return
            await asyncio.sleep(0.02)
        rec.request_shutdown()

    s = asyncio.create_task(stopper())
    try:
        await asyncio.wait_for(rec._run_one(adapter), timeout=5.0)
    finally:
        s.cancel()
        try: await s
        except asyncio.CancelledError: pass

    assert adapter.connect_saw_hook_set, (
        "adapter.on_raw_frame was not a callable when connect() ran — "
        "the recorder must wire the raw-capture hook before calling connect()."
    )


@pytest.mark.asyncio
async def test_handshake_frame_is_persisted_to_raw_store(tmp_path):
    """
    End-to-end: a frame delivered during connect() ends up in the parquet
    raw log. If the hook were wired post-connect, the handshake frame would
    be dropped and the raw partition would be empty.
    """
    rec = _build_recorder(tmp_path)
    adapter = _ConnectCapturingAdapter()

    async def stopper():
        for _ in range(100):
            if adapter.connect_saw_hook_set:
                # Give the writer a moment to flush.
                await asyncio.sleep(0.1)
                rec.request_shutdown()
                return
            await asyncio.sleep(0.02)
        rec.request_shutdown()

    s = asyncio.create_task(stopper())
    try:
        await asyncio.wait_for(rec._run_one(adapter), timeout=5.0)
    finally:
        s.cancel()
        try: await s
        except asyncio.CancelledError: pass

    # Find the raw partition for today and verify the handshake payload landed.
    raw_root = tmp_path / "data" / "raw" / "captest" / "CAP-USD"
    files: List[Path] = list(raw_root.rglob("part-*.parquet"))
    assert files, f"no raw parquet files under {raw_root}"

    import pyarrow.parquet as pq
    rows = []
    for f in files:
        rows.extend(pq.ParquetFile(str(f)).read().to_pylist())

    payloads = [r["payload"] for r in rows]
    assert adapter.handshake_payload in payloads, (
        f"handshake frame missing from raw store; payloads={payloads}"
    )
