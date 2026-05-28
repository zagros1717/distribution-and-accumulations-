/**
 * Hyperliquid Trade Tape Collector
 * --------------------------------
 * Runs every 60s as a scheduled automation.
 * Opens a WebSocket to Hyperliquid, subscribes to `trades` channel
 * for every coin in BotConfig.selected_coins, collects aggressor-side
 * trades for ~45s, then merges them into the rolling TradeTape entity
 * (one record per coin, capped to MAX_TAPE_TRADES).
 *
 * Aggressor side: HL trade messages include `side` ('B' or 'A') indicating
 * which side hit the resting book. 'B' = aggressor buy, 'A' = aggressor sell.
 *
 * This is admin-only (scheduled task).
 */
import { createClientFromRequest } from 'npm:@base44/sdk@0.8.25';

const COLLECT_MS = 40_000;        // collect for 40s, leave 20s for writes
const MAX_TAPE_TRADES = 600;      // ~10 minutes of busy tape, keeps record < 50KB
const TAPE_RETENTION_MS = 15 * 60 * 1000; // drop trades older than 15 min
const MAX_COINS_PERSIST = 40;     // only persist top-N most active coins this burst (matches executor SCAN_LIMIT)
const WRITE_GAP_MS = 250;         // throttle writes to avoid 429s

Deno.serve(async (req) => {
  const start = Date.now();
  try {
    const base44 = createClientFromRequest(req);

    // Permission: only admin (scheduled automation runs as service role,
    // base44.auth.me() returns null in that context — allow when no user).
    const user = await base44.auth.me().catch(() => null);
    if (user && user.role !== 'admin') {
      return Response.json({ error: 'Forbidden' }, { status: 403 });
    }

    const configs = await base44.asServiceRole.entities.BotConfig.list();
    const cfg = configs?.[0];
    const coins = cfg?.selected_coins?.length ? cfg.selected_coins : ['BTC', 'ETH', 'SOL'];

    // Buffer: { coin: [{t, p, s, b}, ...] }
    const buffer = Object.fromEntries(coins.map(c => [c, []]));
    let totalReceived = 0;

    await new Promise((resolve) => {
      const ws = new WebSocket('wss://api.hyperliquid.xyz/ws');
      let closed = false;

      const cleanup = () => {
        if (closed) return;
        closed = true;
        try { ws.close(); } catch { /* ignore */ }
        resolve();
      };

      ws.onopen = () => {
        for (const coin of coins) {
          ws.send(JSON.stringify({
            method: 'subscribe',
            subscription: { type: 'trades', coin },
          }));
        }
      };

      ws.onmessage = (ev) => {
        try {
          const msg = JSON.parse(ev.data);
          if (msg.channel !== 'trades' || !Array.isArray(msg.data)) return;
          for (const tr of msg.data) {
            const coin = tr.coin;
            if (!buffer[coin]) continue;
            const px = parseFloat(tr.px);
            const sz = parseFloat(tr.sz);
            if (!px || !sz) continue;
            buffer[coin].push({
              t: tr.time ?? Date.now(),
              p: px,
              s: sz,
              b: tr.side === 'B',
            });
            totalReceived++;
          }
        } catch { /* ignore malformed */ }
      };

      ws.onerror = () => cleanup();
      ws.onclose = () => cleanup();

      setTimeout(cleanup, COLLECT_MS);
    });

    // Merge into TradeTape entity (one record per coin).
    // Only write coins that received trades this burst — saves API calls + avoids rate limits.
    const existingTapes = await base44.asServiceRole.entities.TradeTape.list('-updated_date', 200);
    const tapeByCoin = Object.fromEntries(existingTapes.map(t => [t.coin, t]));

    const cutoff = Date.now() - TAPE_RETENTION_MS;
    const summary = {};
    const nowIso = new Date().toISOString();

    // Persist only the most active coins this burst (matches executor SCAN_LIMIT).
    // Coins with zero trades AND no existing record are skipped entirely.
    const coinsToWrite = coins
      .filter(c => (buffer[c]?.length ?? 0) > 0 || tapeByCoin[c])
      .sort((a, b) => (buffer[b]?.length ?? 0) - (buffer[a]?.length ?? 0))
      .slice(0, MAX_COINS_PERSIST);

    // Throttled sequential writes (Base44 has per-second rate limits).
    for (const coin of coinsToWrite) {
      const newTrades = buffer[coin] || [];
      const existing = tapeByCoin[coin]?.trades || [];
      const merged = [...existing, ...newTrades]
        .filter(tr => tr.t >= cutoff)
        .slice(-MAX_TAPE_TRADES);

      const payload = {
        coin,
        trades: merged,
        last_collected_at: nowIso,
        burst_trade_count: newTrades.length,
      };

      try {
        if (tapeByCoin[coin]) {
          await base44.asServiceRole.entities.TradeTape.update(tapeByCoin[coin].id, payload);
        } else {
          await base44.asServiceRole.entities.TradeTape.create(payload);
        }
        summary[coin] = { burst: newTrades.length, total: merged.length };
      } catch (e) {
        summary[coin] = { error: e.message?.slice(0, 80) };
      }
      await new Promise(r => setTimeout(r, WRITE_GAP_MS));
    }

    return Response.json({
      ok: true,
      elapsed_ms: Date.now() - start,
      total_received: totalReceived,
      coins: coins.length,
      summary,
    });
  } catch (error) {
    return Response.json({ error: error.message, stack: error.stack }, { status: 500 });
  }
});