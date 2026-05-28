import React, { useEffect, useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { base44 } from '@/api/base44Client';
import { Radar, CheckCircle2, AlertTriangle, TrendingUp, TrendingDown, RefreshCw, ChevronDown, ChevronUp } from 'lucide-react';

const CYCLE_GAP_MS = 30000; // activities within 30s of each other count as one scan cycle

export default function ScanReport() {
  const [activities, setActivities] = useState([]);
  const [loading, setLoading] = useState(true);
  const [expanded, setExpanded] = useState(false);
  const [scanning, setScanning] = useState(false);

  const load = useCallback(async () => {
    const recent = await base44.entities.BotActivity.list('-created_date', 100);
    setActivities(recent);
    setLoading(false);
  }, []);

  const triggerScan = useCallback(async () => {
    setScanning(true);
    try {
      await base44.functions.invoke('botExecutor', {});
      await load();
    } catch (e) {
      console.error('Scan failed:', e.message);
    } finally {
      setScanning(false);
    }
  }, [load]);

  useEffect(() => {
    load();
    let debounceTimer = null;
    const unsub = base44.entities.BotActivity.subscribe(() => {
      if (debounceTimer) clearTimeout(debounceTimer);
      debounceTimer = setTimeout(load, 2000);
    });
    const interval = setInterval(load, 60000);
    return () => {
      unsub();
      clearInterval(interval);
      if (debounceTimer) clearTimeout(debounceTimer);
    };
  }, [load]);

  // Group the most recent contiguous scan cycle
  const buildLatestCycle = () => {
    if (!activities.length) return null;
    const sorted = [...activities].sort((a, b) => new Date(b.created_date) - new Date(a.created_date));
    const cycle = [];
    let lastTs = null;
    for (const a of sorted) {
      const ts = new Date(a.created_date).getTime();
      if (lastTs !== null && lastTs - ts > CYCLE_GAP_MS) break;
      cycle.push(a);
      lastTs = ts;
    }
    return cycle;
  };

  const cycle = buildLatestCycle();

  if (loading) {
    return (
      <div className="glass rounded-xl p-5 text-sm text-muted-foreground flex items-center gap-2">
        <RefreshCw className="w-4 h-4 animate-spin" /> Loading scan report…
      </div>
    );
  }

  if (!cycle || cycle.length === 0) {
    return (
      <div className="glass rounded-xl p-5">
        <div className="flex items-center gap-2 text-sm text-muted-foreground">
          <Radar className="w-4 h-4" />
          No scans yet. Start the bot to begin scanning pairs.
        </div>
      </div>
    );
  }

  const scanned = cycle.filter(a => a.activity_type === 'analysis_complete' && a.coin !== 'SESSION' && a.coin !== 'ALL');
  const signals = cycle.filter(a => a.activity_type === 'signal_found');
  const rejected = cycle.filter(a => a.activity_type === 'signal_rejected');
  const totalPairs = scanned.length + signals.length;
  const lastTs = new Date(cycle[0].created_date);

  const summary = signals.length === 0
    ? `Bot scanned ${totalPairs} pair${totalPairs === 1 ? '' : 's'} and found no signal.`
    : `Bot scanned ${totalPairs} pair${totalPairs === 1 ? '' : 's'} and found ${signals.length} signal${signals.length === 1 ? '' : 's'}.`;

  return (
    <div className="glass rounded-xl overflow-hidden">
      <div className="px-4 py-3 border-b border-border/40 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Radar className="w-4 h-4 text-profit" />
          <h2 className="text-sm font-semibold uppercase tracking-widest text-muted-foreground">Scan Report</h2>
        </div>
        <span className="text-xs text-muted-foreground font-mono">{lastTs.toLocaleTimeString()}</span>
      </div>

      <div className="p-5 space-y-4">
        {/* Headline */}
        <div className={`flex items-start gap-3 p-3 rounded-lg border ${
          signals.length > 0
            ? 'border-profit/30 bg-profit/5'
            : 'border-border/40 bg-muted/20'
        }`}>
          {signals.length > 0
            ? <TrendingUp className="w-5 h-5 text-profit mt-0.5 flex-shrink-0" />
            : <CheckCircle2 className="w-5 h-5 text-muted-foreground mt-0.5 flex-shrink-0" />}
          <div className="flex-1">
            <p className="text-sm font-medium text-foreground">{summary}</p>
            <p className="text-xs text-muted-foreground mt-1 font-mono">
              {scanned.length} no-signal · {signals.length} signal · {rejected.length} rejected
            </p>
          </div>
        </div>

        {/* Signals (always visible if any) */}
        {signals.length > 0 && (
          <div className="space-y-1.5">
            <p className="text-xs uppercase tracking-widest text-muted-foreground font-medium">Signals Found</p>
            {signals.map(s => (
              <div key={s.id} className="flex items-center gap-2 text-xs font-mono p-2 rounded bg-profit/5 border border-profit/20">
                {s.data?.direction === 'long'
                  ? <TrendingUp className="w-3.5 h-3.5 text-profit" />
                  : <TrendingDown className="w-3.5 h-3.5 text-loss" />}
                <span className="font-semibold">{s.coin}</span>
                <span className={s.data?.direction === 'long' ? 'text-profit' : 'text-loss'}>
                  {s.data?.direction?.toUpperCase()}
                </span>
                <span className="text-muted-foreground truncate">{s.data?.reason}</span>
              </div>
            ))}
          </div>
        )}

        {/* Rejections (always visible if any) */}
        {rejected.length > 0 && (
          <div className="space-y-1.5">
            <p className="text-xs uppercase tracking-widest text-muted-foreground font-medium flex items-center gap-1">
              <AlertTriangle className="w-3 h-3" /> Rejected
            </p>
            {rejected.map(r => (
              <div key={r.id} className="text-xs font-mono p-2 rounded bg-loss/5 border border-loss/20">
                <span className="font-semibold">{r.coin}</span>
                <span className="text-muted-foreground ml-2">{r.data?.reason}</span>
              </div>
            ))}
          </div>
        )}

        {/* Manual refresh */}
        <button
          onClick={triggerScan}
          disabled={scanning}
          className="w-full flex items-center justify-center gap-2 text-xs px-3 py-2 rounded-lg border border-profit/30 bg-profit/5 text-profit hover:bg-profit/10 transition-colors disabled:opacity-50 disabled:cursor-not-allowed font-semibold uppercase tracking-widest"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${scanning ? 'animate-spin' : ''}`} />
          {scanning ? 'Scanning all pairs…' : 'Refresh Scan Report'}
        </button>

        {/* Toggle scanned-pair details */}
        {scanned.length > 0 && (
          <div>
            <button
              onClick={() => setExpanded(!expanded)}
              className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground transition-colors"
            >
              {expanded ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
              {expanded ? 'Hide' : 'Show'} scanned pairs ({scanned.length})
            </button>
            <AnimatePresence>
              {expanded && (
                <motion.div
                  initial={{ height: 0, opacity: 0 }}
                  animate={{ height: 'auto', opacity: 1 }}
                  exit={{ height: 0, opacity: 0 }}
                  className="overflow-hidden"
                >
                  <div className="space-y-1 mt-3">
                    {scanned.map(s => (
                      <div key={s.id} className="text-xs font-mono p-2 rounded bg-muted/30 border border-border/30 flex items-baseline gap-2">
                        <span className="font-semibold text-foreground min-w-[60px]">{s.coin}</span>
                        <span className="text-muted-foreground text-[11px] truncate flex-1">{s.data?.reason || s.reason}</span>
                      </div>
                    ))}
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        )}
      </div>
    </div>
  );
}