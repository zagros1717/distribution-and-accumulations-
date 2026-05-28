import React from 'react';
import { TrendingUp, TrendingDown } from 'lucide-react';

export default function LivePrices({ priceHistory, coins }) {
  return (
    <div className="space-y-2 max-h-64 overflow-y-auto">
      {coins.map(coin => {
        const history = priceHistory[coin] || [];
        const current = history[history.length - 1];
        const prev = history[history.length - 2];
        const change = current && prev ? ((current - prev) / prev) * 100 : null;
        const isUp = change >= 0;

        return (
          <div
            key={coin}
            className="flex items-center justify-between px-3 py-2 rounded-lg bg-muted/30 border border-border/40"
          >
            <span className="font-mono font-semibold text-sm">{coin}</span>
            <div className="flex items-center gap-2">
              {current ? (
                <>
                  <span className="font-mono text-sm text-foreground">
                    ${current < 1 ? current.toFixed(5) : current.toFixed(2)}
                  </span>
                  {change !== null && (
                    <div className={`flex items-center gap-0.5 text-xs font-mono ${isUp ? 'text-profit' : 'text-loss'}`}>
                      {isUp ? <TrendingUp className="w-3 h-3" /> : <TrendingDown className="w-3 h-3" />}
                      {isUp ? '+' : ''}{change.toFixed(3)}%
                    </div>
                  )}
                </>
              ) : (
                <span className="text-xs text-muted-foreground">—</span>
              )}
            </div>
          </div>
        );
      })}
      {coins.length === 0 && (
        <p className="text-xs text-muted-foreground text-center py-4">Start bot to see live prices</p>
      )}
    </div>
  );
}