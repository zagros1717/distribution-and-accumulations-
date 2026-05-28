import React from 'react';

export default function OrderBookGauge({ bidVolume, askVolume, coin }) {
  const total = (bidVolume || 0) + (askVolume || 0);
  const bidPct = total > 0 ? Math.round((bidVolume / total) * 100) : 50;
  const askPct = 100 - bidPct;
  const imbalance = total > 0 ? (bidVolume / askVolume).toFixed(2) : '1.00';
  const bullish = bidPct > 55;
  const bearish = askPct > 55;

  return (
    <div className="flex flex-col gap-2">
      <div className="flex justify-between text-xs text-muted-foreground">
        <span>{coin}</span>
        <span className="font-mono">imbalance: <span className={bullish ? 'text-profit' : bearish ? 'text-loss' : 'text-foreground'}>{imbalance}x</span></span>
      </div>
      <div className="flex h-2 rounded-full overflow-hidden gap-0.5">
        <div className="bg-profit rounded-l-full transition-all duration-500" style={{ width: `${bidPct}%` }} />
        <div className="bg-loss rounded-r-full transition-all duration-500" style={{ width: `${askPct}%` }} />
      </div>
      <div className="flex justify-between text-xs font-mono">
        <span className="text-profit">BID {bidPct}%</span>
        <span className="text-loss">ASK {askPct}%</span>
      </div>
    </div>
  );
}