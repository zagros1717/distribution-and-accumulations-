import React, { useState, useEffect, useRef } from 'react';
import { base44 } from '@/api/base44Client';
import { motion, AnimatePresence } from 'framer-motion';
import { TrendingUp, TrendingDown, Zap } from 'lucide-react';

export default function ActivityFeed() {
  const [activities, setActivities] = useState([]);
  const counterRef = useRef(0);

  useEffect(() => {
    // Subscribe to trade updates
    const unsubscribeTrade = base44.entities.Trade.subscribe((event) => {
      const timestamp = new Date();
      let activity = null;

      if (event.type === 'create') {
        activity = {
          id: `activity-${++counterRef.current}`,
          timestamp,
          type: 'open',
          coin: event.data.coin,
          direction: event.data.direction,
          price: event.data.entry_price,
          size: event.data.size_usd,
          reason: event.data.signal_reason,
        };
      } else if (event.type === 'update' && event.data.status === 'closed') {
        activity = {
          id: `activity-${++counterRef.current}`,
          timestamp,
          type: 'close',
          coin: event.data.coin,
          direction: event.data.direction,
          exitPrice: event.data.exit_price,
          pnl: event.data.pnl,
          pnlPct: event.data.pnl_pct,
        };
      }

      if (activity) {
        setActivities(prev => [activity, ...prev].slice(0, 5));
      }
    });

    // Subscribe to bot activity (signal analysis)
    const unsubscribeBot = base44.entities.BotActivity.subscribe((event) => {
      if (event.type === 'create') {
        const activity = {
          id: `activity-${++counterRef.current}`,
          timestamp: new Date(),
          type: event.data.activity_type,
          coin: event.data.coin,
          direction: event.data.direction,
          reason: event.data.reason,
          metrics: event.data.metrics,
        };
        setActivities(prev => [activity, ...prev].slice(0, 5));
      }
    });

    return () => {
      unsubscribeTrade();
      unsubscribeBot();
    };
  }, []);

  if (activities.length === 0) {
    return (
      <div className="text-center py-8 text-muted-foreground text-sm">
        <Zap className="w-5 h-5 mx-auto mb-2 opacity-30" />
        <p>Waiting for bot activity...</p>
      </div>
    );
  }

  return (
    <div className="space-y-2">
      <AnimatePresence>
        {activities.map((activity) => {
          let bgColor = 'bg-muted/30 border-border/50';
          let icon = null;

          if (activity.type === 'open') {
            bgColor = 'bg-profit/5 border-profit/20';
            icon = <TrendingUp className="w-3.5 h-3.5 text-profit flex-shrink-0" />;
          } else if (activity.type === 'close') {
            bgColor = activity.pnl >= 0 ? 'bg-profit/5 border-profit/20' : 'bg-loss/5 border-loss/20';
            icon = <TrendingDown className={`w-3.5 h-3.5 flex-shrink-0 ${activity.pnl >= 0 ? 'text-profit' : 'text-loss'}`} />;
          } else if (activity.type === 'signal_found') {
            bgColor = 'bg-chart-4/10 border-chart-4/30';
            icon = <Zap className="w-3.5 h-3.5 text-chart-4 flex-shrink-0" />;
          } else if (activity.type === 'signal_rejected') {
            bgColor = 'bg-muted/20 border-border/40';
            icon = <div className="w-3.5 h-3.5 rounded-full border border-muted-foreground flex-shrink-0" />;
          } else if (activity.type === 'analysis_complete') {
            bgColor = 'bg-muted/10 border-border/20';
            icon = <div className="w-3.5 h-3.5 rounded-full bg-muted-foreground/30 flex-shrink-0" />;
          }

          return (
            <motion.div
              key={activity.id}
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -10 }}
              className={`flex items-center gap-3 px-3 py-2 rounded-lg text-xs border ${bgColor}`}
            >
              {icon}

              <div className="flex-1 min-w-0">
                {activity.type === 'open' && (
                  <div className="flex items-center gap-2">
                    <span className="font-mono font-semibold">{activity.direction.toUpperCase()}</span>
                    <span className="font-mono">{activity.coin}</span>
                    <span className="text-muted-foreground">@ ${activity.price?.toFixed(2)}</span>
                    <span className="text-muted-foreground text-xs ml-auto flex-shrink-0">
                      {activity.timestamp.toLocaleTimeString()}
                    </span>
                  </div>
                )}
                {activity.type === 'close' && (
                  <div className="flex items-center gap-2">
                    <span className="font-mono">{activity.coin}</span>
                    <span className={activity.pnl >= 0 ? 'text-profit' : 'text-loss'}>
                      {activity.pnl >= 0 ? '+' : ''}${activity.pnl?.toFixed(2)}
                    </span>
                    <span className={`text-xs ${activity.pnl >= 0 ? 'text-profit' : 'text-loss'}`}>
                      ({activity.pnlPct >= 0 ? '+' : ''}{activity.pnlPct?.toFixed(2)}%)
                    </span>
                    <span className="text-muted-foreground text-xs ml-auto flex-shrink-0">
                      {activity.timestamp.toLocaleTimeString()}
                    </span>
                  </div>
                )}
                {activity.type === 'analysis_complete' && (
                  <div className="flex items-center gap-2 text-muted-foreground">
                    <span className="truncate text-xs">{activity.reason}</span>
                    <span className="ml-auto flex-shrink-0">{activity.timestamp.toLocaleTimeString()}</span>
                  </div>
                )}
                {(activity.type === 'signal_found' || activity.type === 'signal_rejected') && (
                  <div className="flex items-center gap-2">
                    <span className="font-mono font-semibold">{activity.coin}</span>
                    <span className={activity.type === 'signal_found' ? 'text-chart-4 font-semibold' : 'text-muted-foreground'}>
                      {activity.type === 'signal_found' ? `${activity.direction.toUpperCase()} signal` : 'No signal'}
                    </span>
                    <span className="text-muted-foreground truncate text-xs ml-auto flex-shrink-0">
                      {activity.reason}
                    </span>
                  </div>
                )}
              </div>
            </motion.div>
          );
        })}
      </AnimatePresence>
    </div>
  );
}