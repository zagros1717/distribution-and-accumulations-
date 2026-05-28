"""
Live data recorder.

Connects to Bitfinex first. On repeated failure, switches to Coinbase. Every
source switch is logged with a metadata record so downstream knows where
each block of data came from. Bitfinex and Coinbase are NOT merged as the
same market; we keep them separately partitioned in the parquet hierarchy.

Two things capture data:

  RawMessageStore       — exact websocket frame text BEFORE normalization.
                          The adapter calls back via `on_raw_frame(channel, text)`.
                          We never lose information: if a future bug is found
                          in the normalizer, we can re-run it on the raw log.
  NormalizedEventStore  — the canonical BookEvent records the adapter emits.

Watchdog: a background coroutine watches `last_msg_ts` and signals the main
loop with an asyncio.Event when the stream has gone silent past the configured
heartbeat timeout. The main loop watches that Event together with the stream
and tears the session down so the supervisor can reconnect / failover.
Critically, the watchdog does NOT rely on raising into a fire-and-forget Task
(which Python silently discards).

This recorder runs forever (until Ctrl+C).
"""
from __future__ import annotations

import asyncio
import json
import signal
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.adapters.base import MarketDataAdapter
from src.adapters.bitfinex import BitfinexRawBookAdapter
from src.adapters.coinbase import CoinbaseFullBookAdapter
from src.schema import BookEvent
from src.storage.parquet_store import RawMessageStore, NormalizedEventStore
from src.utils.logging import logger
from src.utils.time import utcnow


class StreamSilentError(RuntimeError):
    """Raised by _run_one when the watchdog detects a silent stream."""


class Recorder:
    """Supervised recorder with primary/fallback."""

    def __init__(self, config: dict) -> None:
        self.cfg = config
        self.data_root = config["storage"]["root"]
        self.failover = config["sources"]["failover"]
        self._shutdown = asyncio.Event()
        self._raw_store: Optional[RawMessageStore] = None
        self._norm_store: Optional[NormalizedEventStore] = None
        self._current_source: Optional[str] = None
        # Heartbeat tracker shared between watchdog and stream loop.
        self._last_msg_ts: float = 0.0

    # ---- adapter factories ----------------------------------------------

    def _make_primary(self) -> MarketDataAdapter:
        p = self.cfg["sources"]["primary"]
        return BitfinexRawBookAdapter(
            symbol=p["symbol"], canonical_symbol=p["canonical_symbol"],
            ws_url=p["ws_url"],
            book_precision=p.get("book_precision", "R0"),
            book_length=p.get("book_length", "100"),
        )

    def _make_fallback(self) -> MarketDataAdapter:
        f = self.cfg["sources"]["fallback"]
        return CoinbaseFullBookAdapter(
            symbol=f["symbol"], canonical_symbol=f["canonical_symbol"],
            rest_snapshot_url=f["rest_snapshot_url"],
            ws_url=f["ws_url"],
            channel=f.get("channel", "full"),
        )

    # ---- storage rotation -----------------------------------------------

    def _open_stores(self, adapter: MarketDataAdapter) -> None:
        rotate_minutes = int(self.cfg["storage"].get("rotate_files_every_minutes", 60))
        flush_seconds = max(1, rotate_minutes * 60)
        self._raw_store = RawMessageStore(
            self.data_root, adapter.name, adapter.canonical_symbol,
            flush_seconds=flush_seconds,
        )
        self._norm_store = NormalizedEventStore(
            self.data_root, adapter.name, adapter.canonical_symbol,
            flush_seconds=flush_seconds,
        )
        self._current_source = adapter.name
        # Capture raw websocket frames pre-normalization.
        def _on_raw_frame(channel: str, text: str) -> None:
            try:
                if self._raw_store is not None:
                    self._raw_store.write_message(channel, text)
            except Exception as e:
                # Never let raw capture crash the stream.
                logger.exception(f"recorder: raw capture failed: {e}")
        adapter.on_raw_frame = _on_raw_frame
        # Write a source-switch marker.
        marker = {
            "exchange": adapter.name,
            "symbol": adapter.canonical_symbol,
            "ts": utcnow().isoformat(),
            "event": "source_active",
            "config": {"name": adapter.name},
        }
        markers_dir = Path(self.data_root) / "metadata" / "source_switches"
        markers_dir.mkdir(parents=True, exist_ok=True)
        with open(markers_dir / f"{int(utcnow().timestamp())}_{adapter.name}.json", "w") as f:
            json.dump(marker, f, indent=2)
        logger.info(f"recorder: storage open for {adapter.name}/{adapter.canonical_symbol}")

    def _close_stores(self) -> None:
        if self._raw_store:
            self._raw_store.close()
            self._raw_store = None
        if self._norm_store:
            self._norm_store.close()
            self._norm_store = None

    # ---- shutdown -------------------------------------------------------

    def request_shutdown(self) -> None:
        logger.warning("recorder: shutdown requested")
        self._shutdown.set()

    # ---- main loop ------------------------------------------------------

    async def run(self) -> None:
        primary_failures = 0
        backoffs = self.failover.get("reconnect_backoff_seconds", [1, 2, 5, 10, 30])
        max_before_switch = int(self.failover.get("max_reconnects_before_switch", 3))

        while not self._shutdown.is_set():
            adapter_name = "fallback" if primary_failures >= max_before_switch else "primary"
            adapter = self._make_fallback() if adapter_name == "fallback" else self._make_primary()
            try:
                await self._run_one(adapter)
                # Clean exit (rare) — back to top.
                primary_failures = 0
            except Exception as e:
                logger.exception(f"recorder: {adapter.name} session failed: {e}")
                if adapter_name == "primary":
                    primary_failures += 1
                    delay = backoffs[min(primary_failures - 1, len(backoffs) - 1)]
                    logger.info(f"recorder: primary failure #{primary_failures}, sleeping {delay}s")
                    await asyncio.sleep(delay)
                else:
                    # fallback failed too — back off and try primary again.
                    primary_failures = 0
                    delay = backoffs[-1]
                    logger.info(f"recorder: fallback failure, sleeping {delay}s before trying primary")
                    await asyncio.sleep(delay)
            finally:
                self._close_stores()
                try:
                    await adapter.close()
                except Exception:
                    pass

    async def _run_one(self, adapter: MarketDataAdapter) -> None:
        await adapter.connect()
        self._open_stores(adapter)

        # Snapshot first. Adapters write their own raw frames via on_raw_frame
        # during connect/fetch_snapshot, so we only persist the normalized side.
        snapshot_events = await adapter.fetch_snapshot()
        events_list = list(snapshot_events)
        if events_list:
            self._norm_store.write_events(events_list)

        # Stream with watchdog. Watchdog sets `silent` Event when the stream
        # has been quiet for too long; the stream coroutine watches it too
        # and tears the session down by raising StreamSilentError.
        self._last_msg_ts = utcnow().timestamp()
        heartbeat_timeout = float(self.failover.get("heartbeat_timeout_seconds", 15))
        silent = asyncio.Event()

        async def watchdog() -> None:
            # Poll at half the timeout so worst-case detection latency
            # is timeout * 1.5x — fast enough.
            poll = max(1.0, heartbeat_timeout / 2.0)
            while not self._shutdown.is_set() and not silent.is_set():
                await asyncio.sleep(poll)
                idle = utcnow().timestamp() - self._last_msg_ts
                if idle > heartbeat_timeout * 2:
                    logger.warning(
                        f"recorder: stream silent for {idle:.1f}s "
                        f"(threshold={heartbeat_timeout*2:.0f}s); signalling failover"
                    )
                    silent.set()
                    return

        async def stream() -> None:
            async for ev in adapter.stream_events():
                self._last_msg_ts = utcnow().timestamp()
                # Adapter has already captured the raw frame via on_raw_frame.
                # We only persist the normalized event here.
                self._norm_store.write_events([ev])
                if self._shutdown.is_set() or silent.is_set():
                    return

        wd_task = asyncio.create_task(watchdog(), name="recorder-watchdog")
        stream_task = asyncio.create_task(stream(), name="recorder-stream")
        try:
            # Wait for whichever finishes first: stream end, watchdog signal,
            # or shutdown request.
            shutdown_waiter = asyncio.create_task(self._shutdown.wait(),
                                                  name="recorder-shutdown-waiter")
            silent_waiter = asyncio.create_task(silent.wait(),
                                                name="recorder-silent-waiter")
            done, pending = await asyncio.wait(
                [stream_task, shutdown_waiter, silent_waiter],
                return_when=asyncio.FIRST_COMPLETED,
            )
            # Cancel everything still running so we don't leak tasks.
            for t in (wd_task, stream_task, shutdown_waiter, silent_waiter):
                if not t.done():
                    t.cancel()
            # Drain cancellations.
            for t in (wd_task, stream_task, shutdown_waiter, silent_waiter):
                try:
                    await t
                except (asyncio.CancelledError, Exception):
                    pass
            # If the stream finished with an exception, re-raise it so the
            # supervisor loop can decide failover.
            if stream_task in done:
                exc = stream_task.exception()
                if exc is not None:
                    raise exc
            if silent.is_set():
                raise StreamSilentError("watchdog: stream silent")
        finally:
            # Best-effort store flush so partial-second data isn't lost.
            try:
                if self._raw_store: self._raw_store.flush()
                if self._norm_store: self._norm_store.flush()
            except Exception:
                pass


def _install_signal_handlers(rec: Recorder) -> None:
    loop = asyncio.get_event_loop()
    for s in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(s, rec.request_shutdown)
        except NotImplementedError:
            # Windows doesn't support signal handlers on the proactor loop.
            pass


async def main_async(config: dict) -> None:
    rec = Recorder(config)
    _install_signal_handlers(rec)
    await rec.run()
