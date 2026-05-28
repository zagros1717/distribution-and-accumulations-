"""
Record Hyperliquid L2 order book snapshots, aggressor-side trades, funding rates,
and forced liquidations to parquet, partitioned by date and coin. Run as a daemon.

Usage:
    python research/data_recorder/record_l2.py --coins BTC,ETH,SOL --out data/raw

Files written (one per coin per UTC day):
    {out}/trades/coin=BTC/date=2026-05-04.parquet
    {out}/book/coin=BTC/date=2026-05-04.parquet
    {out}/funding/date=2026-05-04.parquet
    {out}/liquidations/date=2026-05-04.parquet
"""
import argparse, asyncio, json, os, time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import pandas as pd
import websockets
import requests

HL_WS = "wss://api.hyperliquid.xyz/ws"
HL_INFO = "https://api.hyperliquid.xyz/info"
BOOK_SNAPSHOT_INTERVAL_S = 1.0   # snapshot the L2 book every second
FLUSH_INTERVAL_S = 30            # flush buffers to parquet every 30s
FUNDING_INTERVAL_S = 300


@dataclass
class TradeRow:
    ts_ms: int; coin: str; px: float; sz: float; is_buy: bool


@dataclass
class BookRow:
    ts_ms: int; coin: str
    bid_px_0: float; bid_sz_0: float; bid_px_1: float; bid_sz_1: float
    bid_px_2: float; bid_sz_2: float; bid_px_3: float; bid_sz_3: float
    bid_px_4: float; bid_sz_4: float
    ask_px_0: float; ask_sz_0: float; ask_px_1: float; ask_sz_1: float
    ask_px_2: float; ask_sz_2: float; ask_px_3: float; ask_sz_3: float
    ask_px_4: float; ask_sz_4: float


@dataclass
class LiqRow:
    ts_ms: int; coin: str; side: str; usd: float; px: float


@dataclass
class FundingRow:
    ts_ms: int; coin: str; funding: float; mark: float


def date_partition(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")


def append_parquet(rows: list, out_dir: Path, partitioning: List[str]) -> None:
    if not rows:
        return
    df = pd.DataFrame([asdict(r) for r in rows])
    if "ts_ms" in df.columns:
        df["date"] = df["ts_ms"].apply(date_partition)
    df.to_parquet(out_dir, partition_cols=partitioning, index=False, engine="pyarrow")


async def collect_trades_and_book(coins: List[str], out: Path):
    trades_buf: List[TradeRow] = []
    book_buf: List[BookRow] = []
    liq_buf: List[LiqRow] = []
    last_flush = time.time()
    last_book_snap = 0.0

    async def flush():
        if trades_buf:
            append_parquet(trades_buf, out / "trades", ["coin", "date"]); trades_buf.clear()
        if book_buf:
            append_parquet(book_buf, out / "book", ["coin", "date"]); book_buf.clear()
        if liq_buf:
            append_parquet(liq_buf, out / "liquidations", ["date"]); liq_buf.clear()

    while True:
        try:
            async with websockets.connect(HL_WS, ping_interval=20) as ws:
                for coin in coins:
                    await ws.send(json.dumps({"method": "subscribe", "subscription": {"type": "trades", "coin": coin}}))
                    await ws.send(json.dumps({"method": "subscribe", "subscription": {"type": "l2Book", "coin": coin}}))
                # HL exposes liquidation events via the `liquidations` channel (per coin)
                for coin in coins:
                    try:
                        await ws.send(json.dumps({"method": "subscribe", "subscription": {"type": "liquidations", "coin": coin}}))
                    except Exception:
                        pass
                async for msg in ws:
                    data = json.loads(msg)
                    ch = data.get("channel")
                    body = data.get("data")
                    now_ms = int(time.time() * 1000)
                    if ch == "trades" and isinstance(body, list):
                        for tr in body:
                            trades_buf.append(TradeRow(
                                ts_ms=int(tr.get("time", now_ms)), coin=tr["coin"],
                                px=float(tr["px"]), sz=float(tr["sz"]), is_buy=(tr.get("side") == "B"),
                            ))
                    elif ch == "l2Book":
                        # Only snapshot once per BOOK_SNAPSHOT_INTERVAL_S to keep file size sane
                        if time.time() - last_book_snap >= BOOK_SNAPSHOT_INTERVAL_S:
                            last_book_snap = time.time()
                            coin = body.get("coin", "")
                            levels = body.get("levels", [[], []])
                            bids = levels[0][:5] if levels else []; asks = levels[1][:5] if len(levels) > 1 else []
                            def at(arr, i, k): return float(arr[i][k]) if i < len(arr) else 0.0
                            book_buf.append(BookRow(
                                ts_ms=now_ms, coin=coin,
                                bid_px_0=at(bids,0,"px"), bid_sz_0=at(bids,0,"sz"),
                                bid_px_1=at(bids,1,"px"), bid_sz_1=at(bids,1,"sz"),
                                bid_px_2=at(bids,2,"px"), bid_sz_2=at(bids,2,"sz"),
                                bid_px_3=at(bids,3,"px"), bid_sz_3=at(bids,3,"sz"),
                                bid_px_4=at(bids,4,"px"), bid_sz_4=at(bids,4,"sz"),
                                ask_px_0=at(asks,0,"px"), ask_sz_0=at(asks,0,"sz"),
                                ask_px_1=at(asks,1,"px"), ask_sz_1=at(asks,1,"sz"),
                                ask_px_2=at(asks,2,"px"), ask_sz_2=at(asks,2,"sz"),
                                ask_px_3=at(asks,3,"px"), ask_sz_3=at(asks,3,"sz"),
                                ask_px_4=at(asks,4,"px"), ask_sz_4=at(asks,4,"sz"),
                            ))
                    elif ch == "liquidations" and isinstance(body, list):
                        for ev in body:
                            usd = float(ev.get("px", 0)) * float(ev.get("sz", 0))
                            liq_buf.append(LiqRow(
                                ts_ms=int(ev.get("time", now_ms)), coin=ev.get("coin", ""),
                                side=("long" if ev.get("side") == "B" else "short"),  # liquidated long = forced sell = appears as buy-aggressor exit; HL convention varies
                                usd=usd, px=float(ev.get("px", 0)),
                            ))
                    if time.time() - last_flush >= FLUSH_INTERVAL_S:
                        await flush(); last_flush = time.time()
        except Exception as e:
            print(f"WS error, reconnecting in 5s: {e}")
            await asyncio.sleep(5)


async def collect_funding(coins: List[str], out: Path):
    while True:
        try:
            data = requests.post(HL_INFO, json={"type": "metaAndAssetCtxs"}, timeout=10).json()
            universe = data[0]["universe"]; ctxs = data[1]
            now_ms = int(time.time() * 1000)
            rows = []
            for i, u in enumerate(universe):
                if u["name"] in coins:
                    rows.append(FundingRow(ts_ms=now_ms, coin=u["name"], funding=float(ctxs[i].get("funding", 0)), mark=float(ctxs[i].get("markPx", 0))))
            if rows:
                append_parquet(rows, out / "funding", ["date"])
        except Exception as e:
            print(f"funding fetch error: {e}")
        await asyncio.sleep(FUNDING_INTERVAL_S)


async def main_async(coins: List[str], out: Path):
    out.mkdir(parents=True, exist_ok=True)
    await asyncio.gather(
        collect_trades_and_book(coins, out),
        collect_funding(coins, out),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--coins", default="BTC,ETH,SOL")
    p.add_argument("--out", default="data/raw")
    args = p.parse_args()
    asyncio.run(main_async([c.strip() for c in args.coins.split(",")], Path(args.out)))
