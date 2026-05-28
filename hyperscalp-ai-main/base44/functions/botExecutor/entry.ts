/**
 * ╔══════════════════════════════════════════════════════════════════╗
 * ║   HYPER-PIPS EXECUTOR v6.0 — Multi-strategy, cost-aware, sized   ║
 * ║                                                                  ║
 * ║  Replaces the v5 single-engine relaxed/moderate/strict modes     ║
 * ║  with three orthogonal strategies, each with its own thesis:     ║
 * ║                                                                  ║
 * ║   • range_mr         — VWAP fade in chopping markets             ║
 * ║   • trend_breakout   — HTF trend continuation w/ CVD confirm     ║
 * ║   • liquidation_fade — fade forced-liquidation cascades          ║
 * ║                                                                  ║
 * ║  New infra:                                                      ║
 * ║   • HTF context (15m + 1h candles, ADX, Choppiness, EMA stack)   ║
 * ║   • Real CVD divergence (not slope)                              ║
 * ║   • Cost-aware filter (E[PnL] >= k * round-trip cost)            ║
 * ║   • Vol-targeted sizing + portfolio vol budget                   ║
 * ║   • Bandit allocator across strategies                           ║
 * ║   • Time stop, partial TP, chandelier trail                      ║
 * ║   • Post-only maker entry option for range_mr                    ║
 * ║                                                                  ║
 * ║  All v5 hardening (signing, IOC fill validation, exchange-side   ║
 * ║  TP/SL triggers, ghost-trade reconciliation, kill switch) is     ║
 * ║  preserved — moved into lib/hl.ts and lib/execution.ts.          ║
 * ╚══════════════════════════════════════════════════════════════════╝
 */
import { createClientFromRequest } from 'npm:@base44/sdk@0.8.25';
import { ethers } from 'npm:ethers@6.11.1';

import { info, getCandles, getOrderBook, getFundingRate, placeIOC, placeTrigger, cancelOrder } from './lib/hl.ts';
import type { Candle, OrderBook } from './lib/hl.ts';
import { atr, type LiquidationEvent, type TapeTrade, btcLeadership } from './lib/features.ts';
import { classifyRegime, type Regime } from './lib/regime.ts';
import { calculateSize, kellyFraction, rollingSharpe } from './lib/sizing.ts';
import { roundTripCostBps, expectedPnlBps, passesEdgeFilter, type CostConfig } from './lib/costs.ts';
import { executeEntry } from './lib/execution.ts';
import { computeWeights, type StrategyName } from './lib/allocator.ts';

import { rangeMR } from './strategies/rangeMR.ts';
import { trendBreakout } from './strategies/trendBreakout.ts';
import { liquidationFade } from './strategies/liquidationFade.ts';
import type { StrategyContext, Signal } from './strategies/types.ts';

const SCAN_LIMIT = 30;
const MIN_TRADE_USD = 12;
const MAX_TRADE_USD = 500;
const COOLDOWN_MS = 10 * 60 * 1000;

// ─── Reconciliation ───────────────────────────────────────────────────────
async function reconcile(base44: any, dbOpenTrades: any[], hlPositions: any[], prices: Record<string, number>) {
  const notes: string[] = [];
  for (const trade of dbOpenTrades) {
    const hlPos = hlPositions.find(p => p.coin === trade.coin);
    if (!hlPos || Math.abs(hlPos.size) < 0.0001) {
      const px = prices[trade.coin] ?? trade.entry_price;
      const pnl = trade.direction === 'long'
        ? (px - trade.entry_price) * (trade.size_usd / trade.entry_price)
        : (trade.entry_price - px) * (trade.size_usd / trade.entry_price);
      await base44.entities.Trade.update(trade.id, {
        status: 'closed', exit_price: px,
        pnl: parseFloat(pnl.toFixed(2)),
        pnl_pct: parseFloat(((pnl / trade.size_usd) * 100).toFixed(2)),
        closed_at: new Date().toISOString(),
        signal_reason: (trade.signal_reason ?? '') + ' [reconciled]',
      });
      notes.push(`Reconciled ghost: ${trade.coin}`);
    }
  }
  return notes;
}

// ─── Per-trade trail / partial-TP / time-stop manager ─────────────────────
async function manageOpenTrade(base44: any, wallet: any, meta: any, trade: any, curPrice: number, c1m: Candle[], cfg: any) {
  const a1 = atr(c1m, 14);

  // Time stop: if trade hasn't progressed >= time_stop_min_progress_r toward target after time_stop_minutes, flatten.
  const ageMin = (Date.now() - new Date(trade.opened_at).getTime()) / 60000;
  const timeStopMin = cfg.risk?.time_stop_minutes ?? 25;
  const minProgressR = cfg.risk?.time_stop_min_progress_r ?? 0.4;
  if (ageMin >= timeStopMin) {
    const r = Math.abs(trade.entry_price - trade.sl_price);
    const progress = trade.direction === 'long'
      ? (curPrice - trade.entry_price) / r
      : (trade.entry_price - curPrice) / r;
    if (progress < minProgressR) {
      // Force-close at market via IOC
      const isBuy = trade.direction === 'short';
      const res = await placeIOC(wallet, meta, trade.coin, isBuy, trade.size_usd, curPrice, true, 25);
      if (res.ok && res.avgPx != null) {
        const exitPx = res.avgPx;
        if (trade.tp_oid) await cancelOrder(wallet, meta, trade.coin, trade.tp_oid);
        if (trade.sl_oid) await cancelOrder(wallet, meta, trade.coin, trade.sl_oid);
        const pnl = trade.direction === 'long'
          ? (exitPx - trade.entry_price) * (trade.size_usd / trade.entry_price)
          : (trade.entry_price - exitPx) * (trade.size_usd / trade.entry_price);
        await base44.entities.Trade.update(trade.id, {
          status: 'closed', exit_price: exitPx,
          pnl: parseFloat(pnl.toFixed(2)),
          pnl_pct: parseFloat(((pnl / trade.size_usd) * 100).toFixed(2)),
          closed_at: new Date().toISOString(),
          signal_reason: (trade.signal_reason ?? '') + ' [time_stop]',
        });
        return { closed: true, reason: 'time_stop' };
      }
    }
  }

  // Partial-TP: when first 1R hit, close partial_tp_fraction at market & move SL to BE
  if (!trade.partial_tp_done && trade.partial_tp_r && trade.partial_tp_fraction) {
    const r = Math.abs(trade.entry_price - trade.sl_price);
    const tpLevel = trade.direction === 'long' ? trade.entry_price + r * trade.partial_tp_r : trade.entry_price - r * trade.partial_tp_r;
    const hit = trade.direction === 'long' ? curPrice >= tpLevel : curPrice <= tpLevel;
    if (hit) {
      const closeSize = trade.size_usd * trade.partial_tp_fraction;
      const isBuy = trade.direction === 'short';
      const res = await placeIOC(wallet, meta, trade.coin, isBuy, closeSize, curPrice, true, 15);
      if (res.ok && res.avgPx != null) {
        const exitPx = res.avgPx;
        const pnlPart = trade.direction === 'long'
          ? (exitPx - trade.entry_price) * (closeSize / trade.entry_price)
          : (trade.entry_price - exitPx) * (closeSize / trade.entry_price);
        // Move SL to break-even
        await base44.entities.Trade.update(trade.id, {
          partial_tp_done: true,
          size_usd: trade.size_usd - closeSize,
          pnl: (trade.pnl ?? 0) + parseFloat(pnlPart.toFixed(2)),
          sl_price: trade.entry_price,
          signal_reason: (trade.signal_reason ?? '') + ` [partial:+$${pnlPart.toFixed(2)}]`,
        });
        return { closed: false, reason: 'partial_tp' };
      }
    }
  }

  // Chandelier trail
  if (trade.trail_type === 'chandelier' && trade.trail_atr_mult > 0 && a1 > 0) {
    const newSL = trade.direction === 'long'
      ? curPrice - a1 * trade.trail_atr_mult
      : curPrice + a1 * trade.trail_atr_mult;
    const better = trade.direction === 'long' ? newSL > trade.sl_price : newSL < trade.sl_price;
    if (better) {
      await base44.entities.Trade.update(trade.id, { sl_price: newSL });
      return { closed: false, reason: 'trail_update' };
    }
  }

  return { closed: false, reason: 'none' };
}

// ─── Liquidation feed ─────────────────────────────────────────────────────
// HL doesn't expose a clean REST liquidation endpoint, but for now we approximate
// via the user-fills public feed in tapeCollector. The executor reads recent
// liquidation events from a `Liquidation` entity if it exists; otherwise returns [].
async function getRecentLiquidations(base44: any, coins: string[]): Promise<LiquidationEvent[]> {
  try {
    const list = await base44.entities.Liquidation?.list?.('-t', 200);
    if (!Array.isArray(list)) return [];
    const cutoff = Date.now() - 30_000;
    return list
      .filter((e: any) => coins.includes(e.coin) && e.t >= cutoff)
      .map((e: any) => ({ t: e.t, coin: e.coin, side: e.side, usd: e.usd } as LiquidationEvent));
  } catch { return []; }
}

// ─── Strategy state I/O ───────────────────────────────────────────────────
async function loadStrategyStates(base44: any, alloc: any, kellyCap: number) {
  const list = (await base44.entities.StrategyState?.list?.('-last_updated', 10).catch(() => [])) ?? [];
  const byName: Record<StrategyName, any> = {} as any;
  for (const s of list) byName[s.strategy as StrategyName] = s;
  const ensure = async (name: StrategyName) => {
    if (byName[name]) return byName[name];
    const created = await base44.entities.StrategyState.create({
      strategy: name, trades_total: 0, trades_window: 0, wins_window: 0,
      pnl_window_usd: 0, fees_window_usd: 0, ret_window: [],
      sharpe: 0, kelly_fraction: 0, weight: 1 / 3,
      last_updated: new Date().toISOString(),
    });
    byName[name] = created;
    return created;
  };
  await ensure('range_mr'); await ensure('trend_breakout'); await ensure('liquidation_fade');
  return byName;
}

async function updateStrategyState(base44: any, name: StrategyName, rMultiple: number, pnlUsd: number, feesUsd: number, lookback: number) {
  const list = await base44.entities.StrategyState.list('-last_updated', 10);
  const cur = list.find((s: any) => s.strategy === name);
  if (!cur) return;
  const ret_window = [...(cur.ret_window ?? []), rMultiple].slice(-lookback);
  const wins_window = (cur.wins_window ?? 0) + (rMultiple > 0 ? 1 : 0);
  const trades_window = ret_window.length;
  const pnl_window_usd = (cur.pnl_window_usd ?? 0) + pnlUsd;
  const fees_window_usd = (cur.fees_window_usd ?? 0) + feesUsd;
  const sharpe = rollingSharpe(ret_window);
  const kf = kellyFraction(ret_window);
  await base44.entities.StrategyState.update(cur.id, {
    trades_total: (cur.trades_total ?? 0) + 1,
    trades_window, wins_window, pnl_window_usd, fees_window_usd, ret_window,
    sharpe, kelly_fraction: kf,
    last_updated: new Date().toISOString(),
  });
}

// ─── Main handler ─────────────────────────────────────────────────────────
Deno.serve(async (req) => {
  const runStart = Date.now();
  const auditLog: string[] = [];
  const log = (msg: string) => { auditLog.push(`[${Date.now() - runStart}ms] ${msg}`); console.log(msg); };

  try {
    const base44 = createClientFromRequest(req);

    const configs = await base44.entities.BotConfig.list();
    const cfg = configs?.[0];
    if (!cfg?.is_active) return Response.json({ status: 'bot_inactive' });
    if (!cfg.wallet_address) return Response.json({ error: 'BotConfig.wallet_address empty' }, { status: 400 });

    let pk = (Deno.env.get('ss') ?? '').trim();
    if (!pk) return Response.json({ error: 'Private key secret "ss" not set' }, { status: 500 });
    if (!pk.startsWith('0x')) pk = '0x' + pk;
    const wallet = new ethers.Wallet(pk);

    const accountAddress = cfg.wallet_address.trim();
    log(`Signer: ${wallet.address} | Account: ${accountAddress}`);

    const [meta, mids, acct, ctxs] = await Promise.all([
      info({ type: 'meta' }),
      info({ type: 'allMids' }),
      info({ type: 'clearinghouseState', user: accountAddress }),
      info({ type: 'metaAndAssetCtxs' }),
    ]);

    const volumeMap: Record<string, number> = {};
    try {
      const universe = ctxs?.[0]?.universe ?? [];
      const cs = ctxs?.[1] ?? [];
      for (let i = 0; i < universe.length; i++) volumeMap[universe[i].name] = parseFloat(cs[i]?.dayNtlVlm ?? 0);
    } catch { /* ignore */ }

    const prices: Record<string, number> = {};
    for (const [k, v] of Object.entries(mids)) prices[k] = parseFloat(v as string);
    const equity = parseFloat(acct?.marginSummary?.accountValue ?? acct?.crossMarginSummary?.accountValue ?? '0');
    log(`Equity: $${equity.toFixed(2)}`);
    if (equity < MIN_TRADE_USD * 2) return Response.json({ status: 'insufficient_equity', equity });

    const hlPositions = (acct?.assetPositions ?? [])
      .filter((p: any) => Math.abs(parseFloat(p.position?.szi ?? 0)) > 0.0001)
      .map((p: any) => ({ coin: p.position.coin, size: parseFloat(p.position.szi), entryPx: parseFloat(p.position.entryPx) }));

    // ── Reconcile ghost trades ────────────────────────────────────────
    const allTrades = await base44.entities.Trade.list('-created_date', 200);
    const openTrades = allTrades.filter((t: any) => t.status === 'open');
    const reconNotes = await reconcile(base44, openTrades, hlPositions, prices);
    if (reconNotes.length) log('Recon: ' + reconNotes.join('; '));

    // ── Drawdown kill switch ──────────────────────────────────────────
    const sessionStartEquity = equity + openTrades.reduce((s: number, t: any) => s - (t.pnl ?? 0), 0);
    const sessionDD = ((sessionStartEquity - equity) / sessionStartEquity) * 100;
    if (sessionDD >= (cfg.risk?.session_drawdown_pct_kill ?? 5)) {
      log(`KILL SWITCH: drawdown ${sessionDD.toFixed(2)}%`);
      await base44.entities.BotActivity.create({ activity_type: 'analysis_complete', coin: 'ALL', reason: `KILL: drawdown ${sessionDD.toFixed(2)}%`, metrics: { equity } });
      return Response.json({ status: 'kill_switch', equity });
    }

    // ── Manage existing positions (trail / partial / time stop / SL/TP) ──
    let closedCount = 0;
    const closedFor: { coin: string; strategy: StrategyName; r: number; pnl: number; fees: number }[] = [];
    for (const trade of openTrades) {
      const cur = prices[trade.coin]; if (!cur) continue;
      const c1m = await getCandles(trade.coin, '1m', 30);

      // Run manager (trail, partial TP, time stop)
      const mgmt = await manageOpenTrade(base44, wallet, meta, trade, cur, c1m, cfg);
      if (mgmt.closed) {
        closedCount++;
        const r = (trade.pnl ?? 0) / Math.max(1, Math.abs(trade.entry_price - trade.sl_price) * (trade.size_usd / trade.entry_price));
        const fees = trade.size_usd * 2 * (cfg.cost_model?.taker_fee_bps ?? 4.5) / 10000;
        closedFor.push({ coin: trade.coin, strategy: trade.strategy, r, pnl: trade.pnl ?? 0, fees });
        continue;
      }

      // Standard SL/TP check (in case exchange-side triggers didn't fire / aren't placed)
      const slHit = trade.sl_price && (trade.direction === 'long' ? cur <= trade.sl_price : cur >= trade.sl_price);
      const tpHit = trade.tp_price && (trade.direction === 'long' ? cur >= trade.tp_price : cur <= trade.tp_price);
      if (slHit || tpHit) {
        const isBuy = trade.direction === 'short';
        const res = await placeIOC(wallet, meta, trade.coin, isBuy, trade.size_usd, cur, true, 20);
        if (!res.ok || res.avgPx == null) { log(`EXIT FAILED ${trade.coin}: ${res.error ?? 'no fill'}`); continue; }
        if (trade.tp_oid) await cancelOrder(wallet, meta, trade.coin, trade.tp_oid);
        if (trade.sl_oid) await cancelOrder(wallet, meta, trade.coin, trade.sl_oid);
        const exitPx = res.avgPx;
        const pnlUsd = trade.direction === 'long'
          ? (exitPx - trade.entry_price) * (trade.size_usd / trade.entry_price)
          : (trade.entry_price - exitPx) * (trade.size_usd / trade.entry_price);
        const fees = trade.size_usd * 2 * (cfg.cost_model?.taker_fee_bps ?? 4.5) / 10000;
        const r = pnlUsd / Math.max(1, Math.abs(trade.entry_price - trade.sl_price) * (trade.size_usd / trade.entry_price));
        await base44.entities.Trade.update(trade.id, {
          status: 'closed', exit_price: exitPx,
          pnl: parseFloat(pnlUsd.toFixed(2)),
          pnl_pct: parseFloat(((pnlUsd / trade.size_usd) * 100).toFixed(2)),
          closed_at: new Date().toISOString(),
          signal_reason: (trade.signal_reason ?? '') + (slHit ? ' [SL]' : ' [TP]'),
        });
        closedCount++;
        if (trade.strategy) closedFor.push({ coin: trade.coin, strategy: trade.strategy, r, pnl: pnlUsd, fees });
      }
    }

    // Update per-strategy rolling stats + Sharpe + Kelly
    const lookback = cfg.allocator?.lookback_trades ?? 50;
    for (const c of closedFor) {
      try { await updateStrategyState(base44, c.strategy, c.r, c.pnl, c.fees, lookback); } catch (e) { /* ignore */ }
    }

    // ── Strategy bandit weights ───────────────────────────────────────
    const states = await loadStrategyStates(base44, cfg.allocator, cfg.allocator?.kelly_cap ?? 0.25);
    const enabled: Record<StrategyName, boolean> = {
      range_mr: !!cfg.strategy_range_mr?.enabled,
      trend_breakout: !!cfg.strategy_trend_breakout?.enabled,
      liquidation_fade: !!cfg.strategy_liquidation_fade?.enabled,
    };
    const weights = computeWeights(
      Object.values(states).map((s: any) => ({ strategy: s.strategy, trades_window: s.trades_window ?? 0, sharpe: s.sharpe ?? 0, kelly_fraction: s.kelly_fraction ?? 0, weight: s.weight ?? 0.33 })),
      enabled, cfg.allocator ?? { use_bandit: true, exploration_eps: 0.1, min_weight: 0.1 },
    );
    log(`Allocator weights: range=${weights.range_mr.toFixed(2)} trend=${weights.trend_breakout.toFixed(2)} liq=${weights.liquidation_fade.toFixed(2)}`);

    // ── Capacity ──────────────────────────────────────────────────────
    const stillOpenCount = openTrades.length - closedCount;
    const slotsAvail = Math.max(0, (cfg.max_open_trades ?? 5) - stillOpenCount);
    if (slotsAvail === 0) return Response.json({ status: 'at_capacity' });

    // ── Cooldown + held coins ─────────────────────────────────────────
    const cutoff = Date.now() - COOLDOWN_MS;
    const cooledCoins = new Set(allTrades.filter((t: any) => t.closed_at && new Date(t.closed_at).getTime() > cutoff).map((t: any) => t.coin));
    const heldCoins = new Set(openTrades.filter((t: any) => t.status === 'open').map((t: any) => t.coin));
    const currentPortfolioVolUsd = openTrades.reduce((s: number, t: any) => s + (t.size_usd * (t.atr_value ?? 0) / Math.max(t.entry_price, 1)), 0);

    // ── Pre-fetch BTC HTF for leadership filter ───────────────────────
    const btc1m = await getCandles('BTC', '1m', 30);

    // ── Pre-fetch liquidations once ───────────────────────────────────
    const allCoins = (cfg.selected_coins ?? []).slice();
    const liquidations = await getRecentLiquidations(base44, allCoins);

    // ── Scan ──────────────────────────────────────────────────────────
    const coinsToScan = allCoins
      .sort((a: string, b: string) => (volumeMap[b] ?? 0) - (volumeMap[a] ?? 0))
      .slice(0, SCAN_LIMIT);
    log(`Scan ${coinsToScan.length} coins, ${slotsAvail} slot(s) available`);

    const tapeRecords = await base44.entities.TradeTape.list('-updated_date', 200);
    const tapeByCoin: Record<string, TapeTrade[]> = Object.fromEntries(tapeRecords.map((t: any) => [t.coin, t.trades || []]));

    const allCandidates: { signal: Signal; coin: string; ob: OrderBook; realizedVol: number; cost: number; sizeUsd: number }[] = [];

    for (const coin of coinsToScan) {
      if (heldCoins.has(coin)) continue;
      if (cooledCoins.has(coin)) continue;
      if (!prices[coin]) continue;

      const [c1m, c5m, c15m, c1h, ob, funding] = await Promise.all([
        getCandles(coin, '1m', 60),
        getCandles(coin, '5m', 60),
        getCandles(coin, '15m', 60),
        getCandles(coin, '1h', 100),
        getOrderBook(coin),
        getFundingRate(coin),
      ]);
      if (c1m.length < 30 || c15m.length < 30 || c1h.length < 50) continue;

      const reg = classifyRegime(c1m, c15m, c1h);
      const btcLead = btcLeadership(btc1m);

      const ctx: StrategyContext = {
        coin, cfg: null,
        c1m, c5m, c15m, c1h, ob, obHistory: [ob], // single snapshot in MVP — could be multi-snap if executor caches them
        tape: tapeByCoin[coin] ?? [],
        liquidations, funding, regime: reg.regime,
        realizedVol: reg.realizedVol, btcLead,
      };

      const candidates = [
        { name: 'range_mr' as const, fn: rangeMR, sub: cfg.strategy_range_mr },
        { name: 'trend_breakout' as const, fn: trendBreakout, sub: cfg.strategy_trend_breakout },
        { name: 'liquidation_fade' as const, fn: liquidationFade, sub: cfg.strategy_liquidation_fade },
      ];

      for (const cand of candidates) {
        if (!enabled[cand.name]) continue;
        if (weights[cand.name] <= 0) continue;
        ctx.cfg = cand.sub;
        const result = cand.fn(ctx);
        if (!result.signal) {
          await base44.entities.BotActivity.create({
            activity_type: 'analysis_complete',
            coin, reason: `${cand.name}: ${result.reject}`,
            metrics: { regime: reg.regime, adx15: reg.adx15, chop15: reg.chop15, htf_stack: reg.htfStack, realized_vol: reg.realizedVol },
          });
          continue;
        }

        const s = result.signal;
        // ── Cost-aware filter ─────────────────────────────────
        const costCfg: CostConfig = cfg.cost_model ?? { taker_fee_bps: 4.5, maker_fee_bps: 1.5, slippage_base_bps: 5, slippage_vol_bps: 15, min_edge_multiplier: 2.0 };
        const entryIsMaker = s.preferredEntryMode === 'post_only';
        const edge = passesEdgeFilter(costCfg, s.entry, s.target, entryIsMaker, false, reg.realizedVol);
        if (!edge.ok) {
          await base44.entities.BotActivity.create({
            activity_type: 'signal_rejected', coin, direction: s.direction,
            reason: `cost-filter: edge=${edge.edgeRatio.toFixed(2)} (need ${costCfg.min_edge_multiplier}); pnl=${edge.pnlBps.toFixed(1)}bps cost=${edge.costBps.toFixed(1)}bps`,
            metrics: edge,
          });
          continue;
        }

        // ── Size ──────────────────────────────────────────────
        const a1 = atr(c1m, 14);
        const atrPct = a1 / Math.max(s.entry, 1);
        const sState = states[s.strategy];
        const sized = calculateSize({
          equity,
          vol_target_usd_per_trade: cfg.vol_target_usd_per_trade ?? 25,
          portfolio_vol_budget_usd: cfg.portfolio_vol_budget_usd ?? 100,
          current_portfolio_vol_usd: currentPortfolioVolUsd,
          max_leverage: cfg.max_leverage ?? 5,
          atr_pct: atrPct,
          expected_holding_minutes: s.expectedHoldingMinutes,
          per_trade_min_usd: MIN_TRADE_USD,
          per_trade_max_usd: MAX_TRADE_USD,
          strategy_kelly_fraction: sState?.kelly_fraction ?? 1,
          allocator_kelly_cap: cfg.allocator?.kelly_cap ?? 0.25,
          bandit_weight: weights[s.strategy],
        });
        if (sized.sizeUsd <= 0) {
          await base44.entities.BotActivity.create({
            activity_type: 'signal_rejected', coin, direction: s.direction,
            reason: `sizing: ${sized.reason}`, metrics: { atr_pct: atrPct },
          });
          continue;
        }

        await base44.entities.BotActivity.create({
          activity_type: 'signal_found', coin, direction: s.direction,
          reason: s.reason,
          metrics: {
            strategy: s.strategy, entry: s.entry, sl: s.stop, tp: s.target,
            size_usd: sized.sizeUsd, edge_ratio: edge.edgeRatio,
            cost_bps: edge.costBps, pnl_bps: edge.pnlBps,
            regime: reg.regime, ...s.features,
          },
        });

        allCandidates.push({ signal: s, coin, ob, realizedVol: reg.realizedVol, cost: edge.costBps, sizeUsd: sized.sizeUsd });
      }
    }

    // Rank candidates by edge ratio × bandit weight
    allCandidates.sort((a, b) => {
      const wa = weights[a.signal.strategy] * (Math.abs(a.signal.target - a.signal.entry) / a.signal.entry);
      const wb = weights[b.signal.strategy] * (Math.abs(b.signal.target - b.signal.entry) / b.signal.entry);
      return wb - wa;
    });

    let opened = 0;
    const executed: any[] = [];
    for (const cand of allCandidates) {
      if (opened >= slotsAvail) break;
      const { signal: s, coin, ob, realizedVol, sizeUsd } = cand;
      const isBuy = s.direction === 'long';

      const order = await executeEntry({
        wallet, meta, coin, isBuy,
        sizeUsd, refPrice: s.entry, ob, realizedVol,
        mode: s.preferredEntryMode,
        postOnlyTimeoutMs: (cfg.strategy_range_mr?.post_only_repost_seconds ?? 30) * 1000,
        slippageBps: 15,
      });
      if (!order.ok || order.avgPx == null || !order.filled) {
        await base44.entities.BotActivity.create({
          activity_type: 'signal_rejected', coin, direction: s.direction,
          reason: `Execute failed: ${order.error ?? 'no fill'}`,
          metrics: { strategy: s.strategy, mode: s.preferredEntryMode },
        });
        continue;
      }

      const actualEntry = order.avgPx;
      const filledCoin = order.filled;
      const filledUsd = parseFloat((filledCoin * actualEntry).toFixed(2));
      const closeIsBuy = !isBuy;

      const [tpRes, slRes] = await Promise.all([
        placeTrigger(wallet, meta, coin, closeIsBuy, filledCoin, s.target, true),
        placeTrigger(wallet, meta, coin, closeIsBuy, filledCoin, s.stop, false),
      ]);

      const a1 = atr(await getCandles(coin, '1m', 30), 14);
      await base44.entities.Trade.create({
        coin,
        strategy: s.strategy,
        direction: s.direction,
        entry_price: actualEntry,
        size_usd: filledUsd,
        tp_price: s.target,
        sl_price: s.stop,
        atr_value: a1,
        status: 'open',
        signal_reason: s.reason + (tpRes.ok && slRes.ok ? ' [protected]' : ' [unprotected]'),
        opened_at: new Date().toISOString(),
        partial_tp_r: s.partialTpR ?? null,
        partial_tp_fraction: s.partialTpFraction ?? null,
        partial_tp_done: false,
        trail_type: s.trailType ?? 'none',
        trail_atr_mult: s.trailAtrMult ?? 0,
        tp_oid: tpRes.ok ? tpRes.oid : null,
        sl_oid: slRes.ok ? slRes.oid : null,
      });
      executed.push({ coin, strategy: s.strategy, direction: s.direction, size: filledUsd, sl: s.stop, tp: s.target, mode: s.preferredEntryMode });
      heldCoins.add(coin);
      opened++;
    }

    const elapsed = Date.now() - runStart;
    log(`Run done: ${elapsed}ms | closed=${closedCount} opened=${opened} | candidates=${allCandidates.length}`);

    await base44.entities.BotActivity.create({
      activity_type: 'analysis_complete', coin: 'SESSION',
      reason: auditLog.join('\n').slice(0, 2000),
      metrics: {
        equity, closed: closedCount, opened, elapsed_ms: elapsed,
        weights, candidates: allCandidates.length,
      },
    });

    return Response.json({
      status: 'executed', account_value: equity,
      closed: closedCount, opened, executed, weights,
      reconciled: reconNotes, elapsed_ms: elapsed,
      timestamp: new Date().toISOString(),
    });
  } catch (error: any) {
    console.error('Bot executor fatal error:', error);
    return Response.json({ error: error.message, stack: error.stack }, { status: 500 });
  }
});
