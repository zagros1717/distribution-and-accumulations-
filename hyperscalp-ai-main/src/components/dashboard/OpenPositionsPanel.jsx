import React, { useState } from 'react';
import { base44 } from '@/api/base44Client';
import { TrendingUp, TrendingDown, X } from 'lucide-react';

export default function OpenPositionsPanel({ positions, prices, onClosed }) {
  const [closingCoin, setClosingCoin] = useState(null);

  const closePosition = async (pos) => {
    const currentPrice = prices?.[pos.coin] || pos.entryPrice;
    const direction = pos.size > 0 ? 'long' : 'short';
    const size_usd = Math.abs(pos.size) * currentPrice;

    setClosingCoin(pos.coin);
    try {
      await base44.functions.invoke('hlTrade', {
        action: 'close',
        coin: pos.coin,
        direction,
        size_usd,
        price: currentPrice,
      });
      if (onClosed) await onClosed();
    } catch (e) {
      alert('Close failed: ' + e.message);
    } finally {
      setClosingCoin(null);
    }
  };

  if (!positions?.length) {
    return (
      <div className="glass rounded-xl p-6 text-center text-sm text-muted-foreground">
        No open positions on Hyperliquid.
      </div>
    );
  }

  return (
    <div className="glass rounded-xl overflow-hidden">
      <div className="px-4 py-3 border-b border-border/40 flex items-center justify-between">
        <h2 className="text-sm font-semibold uppercase tracking-widest text-muted-foreground">
          Open Positions
        </h2>
        <span className="text-xs text-muted-foreground font-mono">{positions.length} live</span>
      </div>
      <div className="grid grid-cols-7 gap-2 px-4 py-2 bg-muted/30 text-xs text-muted-foreground uppercase tracking-widest font-medium">
        <span>Coin</span>
        <span>Side</span>
        <span>Size</span>
        <span>Entry</span>
        <span>Lev</span>
        <span>uPnL</span>
        <span className="text-right">Action</span>
      </div>
      <div>
        {positions.map(pos => {
          const isLong = pos.size > 0;
          const pnlPos = pos.unrealizedPnl >= 0;
          const Icon = isLong ? TrendingUp : TrendingDown;
          return (
            <div key={pos.coin} className="grid grid-cols-7 gap-2 px-4 py-3 border-t border-border/30 text-sm font-mono items-center">
              <span className="font-semibold">{pos.coin}</span>
              <span className={`flex items-center gap-1 ${isLong ? 'text-profit' : 'text-loss'}`}>
                <Icon className="w-3.5 h-3.5" />
                {isLong ? 'LONG' : 'SHORT'}
              </span>
              <span>{Math.abs(pos.size)}</span>
              <span>${pos.entryPrice?.toFixed(2)}</span>
              <span className="text-chart-3">{pos.leverage}×</span>
              <span className={pnlPos ? 'text-profit' : 'text-loss'}>
                {pnlPos ? '+' : ''}${pos.unrealizedPnl?.toFixed(4)}
              </span>
              <div className="flex justify-end">
                <button
                  onClick={() => closePosition(pos)}
                  disabled={closingCoin === pos.coin}
                  className="text-xs px-2 py-1 rounded border border-loss/30 text-loss hover:bg-loss/10 transition-colors disabled:opacity-50 flex items-center gap-1"
                >
                  <X className="w-3 h-3" />
                  {closingCoin === pos.coin ? 'Closing…' : 'Close'}
                </button>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}