import React, { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { base44 } from '@/api/base44Client';
import { Radar, TrendingUp, TrendingDown, Search } from 'lucide-react';
import { formatDistanceToNowStrict } from 'date-fns';

const SCAN_INTERVAL_MS = 15000; // trigger a fresh scan every 15s while bot is active

export default function ScannerPanel({ isActive }) {
  const [activities, setActivities] = useState([]);
  const [lastScan, setLastScan] = useState(null);
  const [scanning, setScanning] = useState(false);

  const loadActivities = useCallback(async () => {
    const recent = await base44.entities.BotActivity.list('-created_date', 30);
    setActivities(recent);
    if (recent[0]) setLastScan(new Date(recent[0].created_date));
  }, []);

  const triggerScan = useCallback(async () => {
    setScanning(true);
    try {
      await base44.functions.invoke('botExecutor', {});
      await loadActivities();
    } catch (e) {
      console.error('scan failed', e.message);
    } finally {
      setScanning(false);
    }
  }, [loadActivities]);

  useEffect(() => {
    loadActivities();
    const refresh = setInterval(loadActivities, 5000);
    return () => clearInterval(refresh);
  }, [loadActivities]);

  // Live scan loop — only when bot is active
  useEffect(() => {
    if (!isActive) return;
    triggerScan();
    const interval = setInterval(triggerScan, SCAN_INTERVAL_MS);
    return () => clearInterval(interval);
  }, [isActive, triggerScan]);

  // Real-time subscription
  useEffect(() => {
    const unsub = base44.entities.BotActivity.subscribe((event) => {
      if (event.type === 'create') {
        setActivities(prev => [event.data, ...prev].slice(0, 30));
        setLastScan(new Date());
      }
    });
    return () => unsub();
  }, []);

  const signalCount = activities.filter(a => a.activity_type === 'signal_found').length;
  const scanCount = activities.length;

  return (
    <div className="glass rounded-xl overflow-hidden">
      <div className="px-4 py-3 border-b border-border/40 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className={`w-2 h-2 rounded-full ${isActive ? 'bg-profit pulse-green' : 'bg-muted-foreground'}`} />
          <h2 className="text-sm font-semibold uppercase tracking-widest text-muted-foreground flex items-center gap-2">
            <Radar className={`w-4 h-4 ${scanning ? 'text-profit animate-spin' : isActive ? 'text-profit' : 'text-muted-foreground'}`} />
            Live Scanner
            {scanning && <span className="text-[10px] text-profit normal-case tracking-normal">scanning…</span>}
          </h2>
        </div>
        <div className="flex items-center gap-3 text-xs font-mono text-muted-foreground">
          <button
            onClick={triggerScan}
            disabled={scanning}
            className="px-2 py-0.5 rounded border border-border hover:bg-muted/50 disabled:opacity-50 transition-colors"
          >
            Scan now
          </button>
          <span>{scanCount} logs</span>
          <span className="text-profit">{signalCount} signals</span>
          {lastScan && (
            <span>· {formatDistanceToNowStrict(lastScan, { addSuffix: true })}</span>
          )}
        </div>
      </div>

      <div className="max-h-72 overflow-y-auto">
        {activities.length === 0 ? (
          <div className="text-center py-12 text-muted-foreground text-sm">
            <Search className="w-8 h-8 mx-auto mb-3 opacity-30" />
            <p>{isActive ? 'Bot starting up — waiting for first scan…' : 'Bot is stopped. Start it to begin scanning.'}</p>
          </div>
        ) : (
          <AnimatePresence initial={false}>
            {activities.map(act => {
              const isSignal = act.activity_type === 'signal_found';
              const isLong = act.direction === 'long';
              const imbalance = act.metrics?.imbalance;
              const confidence = act.metrics?.confidence;
              const threshold = act.metrics?.threshold;

              return (
                <motion.div
                  key={act.id}
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0 }}
                  className={`grid grid-cols-12 gap-2 px-4 py-2 border-t border-border/30 text-xs font-mono items-center ${
                    isSignal ? (isLong ? 'bg-profit/5' : 'bg-loss/5') : ''
                  }`}
                >
                  <span className="col-span-2 font-semibold text-foreground">{act.coin}</span>
                  <span className="col-span-2">
                    {isSignal ? (
                      <span className={`flex items-center gap-1 font-bold ${isLong ? 'text-profit' : 'text-loss'}`}>
                        {isLong ? <TrendingUp className="w-3 h-3" /> : <TrendingDown className="w-3 h-3" />}
                        {isLong ? 'LONG' : 'SHORT'}
                      </span>
                    ) : (
                      <span className="text-muted-foreground">scan</span>
                    )}
                  </span>
                  <span className="col-span-2 text-chart-4">
                    {imbalance != null ? `${imbalance.toFixed(2)}×` : '—'}
                  </span>
                  <span className="col-span-2 text-muted-foreground">
                    thr {threshold != null ? threshold.toFixed(2) : '—'}
                  </span>
                  <span className="col-span-2">
                    {confidence != null && confidence > 0 ? (
                      <span className={isSignal ? 'text-profit' : 'text-muted-foreground'}>
                        {(confidence * 100).toFixed(0)}%
                      </span>
                    ) : (
                      <span className="text-muted-foreground/50">—</span>
                    )}
                  </span>
                  <span className="col-span-2 text-right text-muted-foreground/70">
                    {formatDistanceToNowStrict(new Date(act.created_date), { addSuffix: false })}
                  </span>
                </motion.div>
              );
            })}
          </AnimatePresence>
        )}
      </div>
    </div>
  );
}