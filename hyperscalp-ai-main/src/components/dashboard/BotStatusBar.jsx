import React from 'react';
import { motion } from 'framer-motion';
import { Activity, Wifi, WifiOff } from 'lucide-react';

export default function BotStatusBar({ isActive, tradesCount, lastSignal }) {
  return (
    <div className="flex items-center gap-3 px-4 py-2 glass rounded-lg text-xs font-mono">
      <div className="flex items-center gap-2">
        {isActive ? (
          <span className="w-2 h-2 rounded-full bg-profit pulse-green inline-block" />
        ) : (
          <span className="w-2 h-2 rounded-full bg-muted-foreground inline-block" />
        )}
        <span className={isActive ? 'text-profit' : 'text-muted-foreground'}>
          {isActive ? 'BOT ACTIVE' : 'BOT STOPPED'}
        </span>
      </div>
      <span className="text-border">|</span>
      <div className="flex items-center gap-1 text-muted-foreground">
        <Activity className="w-3 h-3" />
        <span>{tradesCount} trades today</span>
      </div>
      {lastSignal && (
        <>
          <span className="text-border">|</span>
          <span className="text-muted-foreground truncate max-w-[200px]">Last: {lastSignal}</span>
        </>
      )}
      <span className="text-border">|</span>
      <div className="flex items-center gap-1 text-profit">
        <Wifi className="w-3 h-3" />
        <span>Hyperliquid</span>
      </div>
    </div>
  );
}