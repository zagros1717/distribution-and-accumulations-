import React, { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { base44 } from '@/api/base44Client';
import { Play, Square, Settings, TrendingUp, Zap, DollarSign, BarChart2, RefreshCw } from 'lucide-react';
import SignalPopup from '@/components/dashboard/SignalPopup';
import { Button } from '@/components/ui/button';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import StatCard from '@/components/dashboard/StatCard';
import BotStatusBar from '@/components/dashboard/BotStatusBar';
import TradeRow from '@/components/dashboard/TradeRow';
import LivePrices from '@/components/dashboard/LivePrices';
import OrderBookGauge from '@/components/dashboard/OrderBookGauge';
import BotSettingsPanel from '@/components/settings/BotSettingsPanel';
import OpenPositionsPanel from '@/components/dashboard/OpenPositionsPanel';
import ScanReport from '@/components/dashboard/ScanReport';

const REFRESH_INTERVAL = 60000; // 60s — display refresh

export default function Dashboard() {
  const [config, setConfig] = useState(null);
  const [trades, setTrades] = useState([]);
  const [orderBooks, setOrderBooks] = useState({});
  const [priceHistory, setPriceHistory] = useState({});
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [isActive, setIsActive] = useState(false);
  const [accountValue, setAccountValue] = useState(null);
  const [hlPositions, setHlPositions] = useState([]);
  const [allHlCoins, setAllHlCoins] = useState([]);
  const [lastRefresh, setLastRefresh] = useState(null);
  const [signalPopups, setSignalPopups] = useState([]);

  // ── Signal popup subscription ────────────────────────────────────────────
  useEffect(() => {
    const unsub = base44.entities.BotActivity.subscribe((event) => {
      if (event.type === 'create' && event.data?.activity_type === 'signal_found') {
        const popup = {
          id: `signal-${Date.now()}`,
          coin: event.data.coin,
          direction: event.data.direction,
          reason: event.data.reason,
          metrics: event.data.metrics,
        };
        setSignalPopups(prev => [popup, ...prev].slice(0, 2));
        setTimeout(() => {
          setSignalPopups(prev => prev.filter(p => p.id !== popup.id));
        }, 5000);
      }
    });
    return () => unsub();
  }, []);

  // ── Load config + trades from DB ─────────────────────────────────────────
  const loadData = useCallback(async () => {
    const [configs, allTrades] = await Promise.all([
      base44.entities.BotConfig.list(),
      base44.entities.Trade.list('-created_date', 100),
    ]);
    if (configs.length > 0) {
      setConfig(configs[0]);
      setIsActive(configs[0].is_active || false);
    }
    setTrades(allTrades);
  }, []);

  // ── Load prices + order books for display ────────────────────────────────
  const loadPrices = useCallback(async (coins) => {
    if (!coins?.length) return;
    try {
      const priceRes = await base44.functions.invoke('hlPrice', { coins });
      const prices = priceRes.data?.prices || {};
      if (priceRes.data?.allCoins?.length) setAllHlCoins(priceRes.data.allCoins);

      setPriceHistory(prev => {
        const next = { ...prev };
        coins.forEach(coin => {
          if (!prices[coin]) return;
          const hist = prev[coin] || [];
          next[coin] = [...hist.slice(-20), prices[coin]];
        });
        return next;
      });

      // Fetch order books for display (first 4 coins)
      const displayCoins = coins.slice(0, 4);
      const newOrderBooks = {};
      for (const coin of displayCoins) {
        if (!prices[coin]) continue;
        try {
          const obRes = await base44.functions.invoke('hlOrderBook', { coin });
          if (obRes.data) newOrderBooks[coin] = { bidVolume: obRes.data.bidVolume, askVolume: obRes.data.askVolume };
        } catch (e) { /* ignore */ }
        await new Promise(r => setTimeout(r, 150));
      }
      setOrderBooks(newOrderBooks);
    } catch (e) {
      console.error('Price fetch failed:', e.message);
    }
  }, []);

  // ── Load account balance ─────────────────────────────────────────────────
  const loadAccount = useCallback(async () => {
    try {
      const res = await base44.functions.invoke('hlAccount', {});
      if (res.data?.accountValue != null) {
        setAccountValue(res.data.accountValue);
        setHlPositions(res.data.positions || []);
      }
    } catch (e) {
      console.error('loadAccount failed:', e.message);
    }
  }, []);

  // ── Initial load ─────────────────────────────────────────────────────────
  useEffect(() => {
    loadData();
    loadAccount();
  }, [loadData, loadAccount]);

  // ── Periodic display refresh (prices only, no trading logic) ─────────────
  useEffect(() => {
    const coins = config?.selected_coins?.length ? config.selected_coins : ['BTC', 'ETH', 'SOL', 'AVAX', 'DOGE'];
    loadPrices(coins);
    const interval = setInterval(async () => {
      loadPrices(coins);
      const freshTrades = await base44.entities.Trade.list('-created_date', 100);
      setTrades(freshTrades);
      setLastRefresh(new Date());
    }, REFRESH_INTERVAL);
    return () => clearInterval(interval);
  }, [config, loadPrices]);

  // ── Toggle bot (just flips is_active in DB — backend automation handles the rest) ──
  const toggleBot = async () => {
    const newState = !isActive;
    setIsActive(newState);
    if (config?.id) {
      await base44.entities.BotConfig.update(config.id, { is_active: newState });
    }
  };

  // ── Manual close (calls backend to close on exchange + update DB) ────────
  const closePositionManually = async (trade) => {
    const currentPrice = (priceHistory[trade.coin] || []).at(-1);
    if (!currentPrice) { alert('No price available'); return; }

    await base44.functions.invoke('hlTrade', {
      action: 'close',
      coin: trade.coin,
      direction: trade.direction,
      size_usd: trade.size_usd,
      price: currentPrice,
    });

    const pnl = trade.direction === 'long'
      ? parseFloat(((currentPrice - trade.entry_price) / trade.entry_price * trade.size_usd).toFixed(2))
      : parseFloat(((trade.entry_price - currentPrice) / trade.entry_price * trade.size_usd).toFixed(2));

    await base44.entities.Trade.update(trade.id, {
      status: 'closed',
      exit_price: currentPrice,
      pnl,
      pnl_pct: parseFloat(((pnl / trade.size_usd) * 100).toFixed(3)),
      closed_at: new Date().toISOString(),
    });

    const fresh = await base44.entities.Trade.list('-created_date', 100);
    setTrades(fresh);
    loadAccount();
  };

  // ── Stats ────────────────────────────────────────────────────────────────
  const closedTrades = trades.filter(t => t.status === 'closed');
  const openTrades = trades.filter(t => t.status === 'open');
  const todayTrades = trades.filter(t => new Date(t.created_date).toDateString() === new Date().toDateString());
  const totalPnl = closedTrades.reduce((sum, t) => sum + (t.pnl || 0), 0);
  const winCount = closedTrades.filter(t => t.pnl > 0).length;
  const winRate = closedTrades.length > 0 ? Math.round((winCount / closedTrades.length) * 100) : 0;
  const configCoins = config?.selected_coins?.length ? config.selected_coins : (allHlCoins.length ? allHlCoins : ['BTC', 'ETH', 'SOL', 'AVAX', 'DOGE']);

  return (
    <div className="min-h-screen bg-background font-inter">
      {/* Header */}
      <div className="border-b border-border/50 px-6 py-4">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-profit/10 border border-profit/30 flex items-center justify-center">
              <Zap className="w-4 h-4 text-profit" />
            </div>
            <div>
              <h1 className="text-lg font-semibold tracking-tight">HyperScalper</h1>
              <p className="text-xs text-muted-foreground">
                Live Trading · Hyperliquid DEX ·{' '}
                <span className="text-chart-3 font-mono font-semibold">3× Perp</span>
                {lastRefresh && (
                  <span className="ml-2 text-muted-foreground/60">· refreshed {lastRefresh.toLocaleTimeString()}</span>
                )}
              </p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <BotStatusBar isActive={isActive} tradesCount={todayTrades.length} lastSignal={null} />
            <Button
              variant="ghost"
              size="icon"
              onClick={() => setSettingsOpen(true)}
              className="text-muted-foreground hover:text-foreground"
            >
              <Settings className="w-4 h-4" />
            </Button>
            <Button
              onClick={toggleBot}
              className={`gap-2 font-semibold transition-all ${
                isActive
                  ? 'bg-loss/10 text-loss border border-loss/30 hover:bg-loss/20'
                  : 'bg-profit text-primary-foreground hover:bg-profit/90 glow-green'
              }`}
            >
              {isActive ? <><Square className="w-4 h-4" />Stop Bot</> : <><Play className="w-4 h-4" />Start Bot</>}
            </Button>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-6 py-6 space-y-6">
        {/* Bot running indicator */}
        {isActive && (
          <div className="flex items-center gap-2 text-xs text-profit bg-profit/5 border border-profit/20 rounded-lg px-4 py-2">
            <div className="w-2 h-2 rounded-full bg-profit pulse-green" />
            Bot is running — trades execute automatically.
          </div>
        )}

        {/* Stats row */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="glass rounded-xl p-5 flex flex-col gap-2">
            <div className="flex items-center justify-between">
              <span className="text-xs text-muted-foreground uppercase tracking-widest">Account Balance</span>
              <button onClick={loadAccount} className="text-muted-foreground hover:text-foreground transition-colors">
                <RefreshCw className="w-3.5 h-3.5" />
              </button>
            </div>
            <div className="text-2xl font-semibold font-mono text-foreground">
              {accountValue != null ? `$${accountValue.toFixed(2)}` : '—'}
            </div>
            <div className="text-xs text-muted-foreground">Hyperliquid USDC</div>
          </div>
          <StatCard
            label="Realised PnL"
            value={`${totalPnl >= 0 ? '+' : ''}$${totalPnl.toFixed(2)}`}
            positive={totalPnl > 0}
            negative={totalPnl < 0}
            icon={DollarSign}
            mono
          />
          <StatCard
            label="Win Rate"
            value={`${winRate}%`}
            positive={winRate > 50}
            negative={winRate < 40}
            sub={`${winCount}/${closedTrades.length} trades`}
            icon={TrendingUp}
            mono
          />
          <StatCard
            label="Open Positions"
            value={hlPositions.length}
            sub={`max ${config?.max_open_trades || 20}`}
            icon={BarChart2}
            mono
          />
        </div>

        {/* Chart + Order Books + Activity */}
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-4">
          <div className="lg:col-span-2 glass rounded-xl p-5">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-sm font-semibold uppercase tracking-widest text-muted-foreground">Live Prices</h2>
              <span className="text-xs text-muted-foreground font-mono">{configCoins.length} pairs</span>
            </div>
            <LivePrices priceHistory={priceHistory} coins={configCoins} />
          </div>

          <div className="glass rounded-xl p-5 space-y-4">
            <h2 className="text-sm font-semibold uppercase tracking-widest text-muted-foreground">Order Book Imbalance</h2>
            {configCoins.slice(0, 4).map(coin => (
              <OrderBookGauge
                key={coin}
                coin={coin}
                bidVolume={orderBooks[coin]?.bidVolume || 0}
                askVolume={orderBooks[coin]?.askVolume || 0}
              />
            ))}
          </div>


        </div>

        {/* Scan Report */}
        <ScanReport />

        {/* Live HL Positions */}
        <OpenPositionsPanel
          positions={hlPositions}
          prices={Object.fromEntries(Object.entries(priceHistory).map(([k, v]) => [k, v.at(-1)]))}
          onClosed={loadAccount}
        />

      </div>

      {/* Settings Dialog */}
      <Dialog open={settingsOpen} onOpenChange={setSettingsOpen}>
        <DialogContent className="max-w-2xl bg-card border-border max-h-[90vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle className="text-base font-semibold tracking-tight">Bot Configuration</DialogTitle>
          </DialogHeader>
          <BotSettingsPanel
            config={config}
            onSaved={() => {
              setSettingsOpen(false);
              loadData();
            }}
          />
        </DialogContent>
      </Dialog>

      {/* Signal Popups */}
      <div className="fixed bottom-6 right-6 z-50 flex flex-col gap-2">
        <AnimatePresence>
          {signalPopups.map(popup => (
            <SignalPopup key={popup.id} popup={popup} onDismiss={() => setSignalPopups(prev => prev.filter(p => p.id !== popup.id))} />
          ))}
        </AnimatePresence>
      </div>
    </div>
  );
}