import React from 'react';
import { TrendingUp, TrendingDown, X } from 'lucide-react';
import { format } from 'date-fns';
import { Button } from '@/components/ui/button';

export default function TradeRow({ trade, onClose }) {
  const isProfit = trade.pnl >= 0;
  const isLong = trade.direction === 'long';

  return (
    <div className="grid grid-cols-8 gap-2 items-center px-4 py-3 border-b border-border/40 hover:bg-white/[0.02] transition-colors text-sm">
      <div className="flex items-center gap-2 font-semibold">
        <span className={`text-xs px-1.5 py-0.5 rounded font-mono font-bold ${isLong ? 'bg-profit/10 text-profit' : 'bg-loss/10 text-loss'}`}>
          {isLong ? 'L' : 'S'}
        </span>
        {trade.coin}
      </div>
      <div className="font-mono text-xs text-muted-foreground">
        ${trade.entry_price?.toFixed(2)}
      </div>
      <div className="font-mono text-xs space-y-0.5">
        <div className="text-profit">${trade.tp_price?.toFixed(2) || '—'}</div>
        <div className="text-loss">${trade.sl_price?.toFixed(2) || '—'}</div>
      </div>
      <div className="font-mono text-xs text-muted-foreground">
        {trade.exit_price ? `$${trade.exit_price.toFixed(2)}` : '—'}
      </div>
      <div className="font-mono text-xs">
        ${trade.size_usd?.toFixed(0)}
      </div>
      <div className={`font-mono text-xs font-semibold ${isProfit ? 'text-profit' : 'text-loss'}`}>
        {trade.pnl != null ? `${isProfit ? '+' : ''}$${trade.pnl.toFixed(2)}` : '—'}
      </div>
      <div className={`font-mono text-xs ${isProfit ? 'text-profit' : 'text-loss'}`}>
        {trade.pnl_pct != null ? `${isProfit ? '+' : ''}${trade.pnl_pct.toFixed(2)}%` : '—'}
      </div>
      <div className="text-xs text-muted-foreground">
        {trade.status === 'open' ? (
          <span className="text-chart-4">open</span>
        ) : (
          trade.closed_at ? format(new Date(trade.closed_at), 'HH:mm:ss') : '—'
        )}
      </div>
      {trade.status === 'open' && onClose && (
        <Button
          size="sm"
          variant="ghost"
          onClick={() => onClose(trade)}
          className="h-6 w-6 p-0 text-muted-foreground hover:text-loss hover:bg-loss/10"
        >
          <X className="w-3 h-3" />
        </Button>
      )}
    </div>
  );
}