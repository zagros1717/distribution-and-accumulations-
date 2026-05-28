import React from 'react';
import { motion } from 'framer-motion';

export default function StatCard({ label, value, sub, positive, negative, icon: Icon, mono }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      className="glass rounded-xl p-5 flex flex-col gap-2"
    >
      <div className="flex items-center justify-between">
        <span className="text-xs text-muted-foreground uppercase tracking-widest">{label}</span>
        {Icon && <Icon className="w-4 h-4 text-muted-foreground" />}
      </div>
      <div className={`text-2xl font-semibold ${mono ? 'font-mono' : ''} ${positive ? 'text-profit' : negative ? 'text-loss' : 'text-foreground'}`}>
        {value}
      </div>
      {sub && <div className="text-xs text-muted-foreground">{sub}</div>}
    </motion.div>
  );
}