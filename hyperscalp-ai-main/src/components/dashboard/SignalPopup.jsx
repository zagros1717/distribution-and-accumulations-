import { motion } from 'framer-motion';
import { TrendingUp, TrendingDown, X } from 'lucide-react';

export default function SignalPopup({ popup, onDismiss }) {
  const isLong = popup.direction === 'long';

  return (
    <motion.div
      initial={{ opacity: 0, x: 60, scale: 0.95 }}
      animate={{ opacity: 1, x: 0, scale: 1 }}
      exit={{ opacity: 0, x: 60, scale: 0.95 }}
      transition={{ type: 'spring', stiffness: 300, damping: 25 }}
      className={`relative flex items-start gap-3 px-4 py-3 rounded-xl border shadow-lg w-72 ${
        isLong
          ? 'bg-profit/10 border-profit/40 glow-green'
          : 'bg-loss/10 border-loss/40 glow-red'
      }`}
    >
      <div className={`mt-0.5 p-1.5 rounded-lg ${isLong ? 'bg-profit/20' : 'bg-loss/20'}`}>
        {isLong
          ? <TrendingUp className="w-4 h-4 text-profit" />
          : <TrendingDown className="w-4 h-4 text-loss" />
        }
      </div>

      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <span className={`text-xs font-bold uppercase tracking-widest ${isLong ? 'text-profit' : 'text-loss'}`}>
            {isLong ? 'LONG' : 'SHORT'} Signal
          </span>
          <span className="text-sm font-mono font-semibold text-foreground">{popup.coin}</span>
        </div>
        <p className="text-xs text-muted-foreground mt-0.5 truncate">{popup.reason}</p>
        {popup.metrics?.confidence != null && (
          <div className="mt-1.5 flex items-center gap-2">
            <div className="flex-1 h-1 bg-muted rounded-full overflow-hidden">
              <div
                className={`h-full rounded-full ${isLong ? 'bg-profit' : 'bg-loss'}`}
                style={{ width: `${Math.min(popup.metrics.confidence * 100, 100)}%` }}
              />
            </div>
            <span className="text-xs font-mono text-muted-foreground">
              {(popup.metrics.confidence * 100).toFixed(0)}% conf
            </span>
          </div>
        )}
      </div>

      <button onClick={onDismiss} className="text-muted-foreground hover:text-foreground transition-colors mt-0.5">
        <X className="w-3.5 h-3.5" />
      </button>
    </motion.div>
  );
}