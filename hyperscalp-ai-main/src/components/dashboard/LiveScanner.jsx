import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { base44 } from '@/api/base44Client';
import { Radar, TrendingUp, TrendingDown, Loader2, CheckCircle2 } from 'lucide-react';

const SCAN_DELAY_MS = 800; // delay between coins so user can see the scan happening

export default function LiveScanner({ isActive, coins, threshold = 1.3 }) {
  const [scanState, setScanState] = useState({}); // { COIN: { status, ratio, decision, ts } }
  const [currentCoin, setCurrentCoin] = useState(null);
  const [cycleCount, setCycleCount] = useState(0);
  const stopRef = useRef(false);

  useEffect(() => {
    if (!isActive || !coins?.length) {
      stopRef.current = true;
      setCurrentCoin(null);
      return;
    }
    stopRef.current = false;

    const scanLoop = async () => {
      while (!stopRef.current) {
        for (const coin of coins) {
          if (stopRef.current) break;
          setCurrentCoin(coin);
          setScanState(prev => ({ ...prev, [coin]: { ...prev[coin], status: 'scanning' } }));

          try {
            const res = await base44.functions.invoke('hlOrderBook', { coin });
            const bidVol = res.data?.bidVolume || 0;
            const askVol = res.data?.askVolume || 0;
            const ratio = askVol > 0 ? bidVol / askVol : 1;

            let decision = null;
            if (ratio > threshold) decision = 'long';
            else if (ratio < 1 / threshold) decision = 'short';

            setScanState(prev => ({
              ...prev,
              [coin]: { status: 'done', ratio, bidVol, askVol, decision, ts: Date.now() },
            }));
          } catch (e) {
            setScanState(prev => ({ ...prev, [coin]: { status: 'error', ts: Date.now() } }));
          }

          await new Promise(r => setTimeout(r, SCAN_DELAY_MS));
        }
        if (!stopRef.current) setCycleCount(c => c + 1);
      }
    };

    scanLoop();
    return () => { stopRef.current = true; };
  }, [isActive, coins, threshold]);

  const signalCount = Object.values(scanState).filter(s => s?.decision).length;

  return (
    <div className="glass rounded-xl overflow-hidden">
      <div className="px-4 py-3 border-b border-border/40 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Radar className={`w-4 h-4 ${isActive ? 'text-profit animate-pulse' : 'text-muted-foreground'}`} />
          <h2 className="text-sm font-semibold uppercase tracking-widest text-muted-foreground">
            Live Scanner
          </h2>
          {currentCoin && (
            <span className="text-xs text-profit font-mono flex items-center gap-1">
              <Loader2 className="w-3 h-3 animate-spin" />
              checking {currentCoin}…
            </span>
          )}
        </div>
        <div className="flex items-center gap-3 text-xs font-mono text-muted-foreground">
          <span>cycle #{cycleCount}</span>
          <span>thr {threshold.toFixed(2)}×</span>
          <span className="text-profit">{signalCount} signals</span>
        </div>
      </div>

      {!isActive ? (
        <div className="text-center py-12 text-muted-foreground text-sm">
          <Radar className="w-8 h-8 mx-auto mb-3 opacity-30" />
          <p>Bot is stopped — start it to begin live scanning.</p>
        </div>
      ) : (
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-2 p-3">
          <AnimatePresence>
            {coins.map(coin => {
              const s = scanState[coin] || { status: 'pending' };
              const isCurrent = currentCoin === coin;
              const hasSignal = !!s.decision;
              const isLong = s.decision === 'long';

              return (
                <motion.div
                  key={coin}
                  layout
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  className={`rounded-lg border p-3 transition-all ${
                    isCurrent
                      ? 'border-profit bg-profit/10 glow-green'
                      : hasSignal
                        ? isLong ? 'border-profit/40 bg-profit/5' : 'border-loss/40 bg-loss/5'
                        : 'border-border/40 bg-muted/20'
                  }`}
                >
                  <div className="flex items-center justify-between mb-1.5">
                    <span className="font-mono font-semibold text-sm">{coin}</span>
                    {s.status === 'scanning' && <Loader2 className="w-3 h-3 animate-spin text-profit" />}
                    {s.status === 'done' && !hasSignal && <CheckCircle2 className="w-3 h-3 text-muted-foreground/60" />}
                    {hasSignal && (isLong
                      ? <TrendingUp className="w-3 h-3 text-profit" />
                      : <TrendingDown className="w-3 h-3 text-loss" />
                    )}
                  </div>

                  {s.status === 'pending' && (
                    <div className="text-xs text-muted-foreground/50 font-mono">waiting…</div>
                  )}
                  {s.status === 'error' && (
                    <div className="text-xs text-loss font-mono">error</div>
                  )}
                  {s.status !== 'pending' && s.status !== 'error' && s.ratio != null && (
                    <>
                      <div className={`text-base font-mono font-bold ${
                        hasSignal ? (isLong ? 'text-profit' : 'text-loss') : 'text-foreground'
                      }`}>
                        {s.ratio.toFixed(2)}×
                      </div>
                      {/* Mini bid/ask bar */}
                      <div className="mt-1.5 h-1 rounded-full overflow-hidden bg-muted flex">
                        <div
                          className="bg-profit"
                          style={{ width: `${(s.bidVol / (s.bidVol + s.askVol)) * 100}%` }}
                        />
                        <div
                          className="bg-loss"
                          style={{ width: `${(s.askVol / (s.bidVol + s.askVol)) * 100}%` }}
                        />
                      </div>
                      {hasSignal && (
                        <div className={`text-[10px] font-bold uppercase tracking-wider mt-1 ${
                          isLong ? 'text-profit' : 'text-loss'
                        }`}>
                          {s.decision} signal
                        </div>
                      )}
                    </>
                  )}
                </motion.div>
              );
            })}
          </AnimatePresence>
        </div>
      )}
    </div>
  );
}