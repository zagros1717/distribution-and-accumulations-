import React from 'react';
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';

const CustomTooltip = ({ active, payload, label }) => {
  if (active && payload && payload.length) {
    const val = payload[0].value;
    return (
      <div className="glass rounded-lg px-3 py-2 text-xs font-mono">
        <div className="text-muted-foreground mb-1">{label}</div>
        <div className={val >= 0 ? 'text-profit' : 'text-loss'}>
          {val >= 0 ? '+' : ''}${val.toFixed(2)}
        </div>
      </div>
    );
  }
  return null;
};

export default function PnlChart({ trades }) {
  // Build cumulative PnL from closed trades
  let cumulative = 0;
  const data = trades
    .filter(t => t.status === 'closed' && t.pnl != null)
    .sort((a, b) => new Date(a.closed_at) - new Date(b.closed_at))
    .map((t, i) => {
      cumulative += t.pnl;
      return {
        name: `T${i + 1}`,
        pnl: parseFloat(cumulative.toFixed(2)),
      };
    });

  if (data.length === 0) {
    data.push({ name: 'Start', pnl: 0 });
  }

  const isPositive = data[data.length - 1]?.pnl >= 0;
  const color = isPositive ? '#22c55e' : '#ef4444';

  return (
    <ResponsiveContainer width="100%" height={160}>
      <AreaChart data={data} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
        <defs>
          <linearGradient id="pnlGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={color} stopOpacity={0.25} />
            <stop offset="100%" stopColor={color} stopOpacity={0.01} />
          </linearGradient>
        </defs>
        <XAxis dataKey="name" tick={{ fill: '#64748b', fontSize: 10 }} axisLine={false} tickLine={false} />
        <YAxis tick={{ fill: '#64748b', fontSize: 10 }} axisLine={false} tickLine={false} width={50} tickFormatter={v => `$${v}`} />
        <Tooltip content={<CustomTooltip />} />
        <Area type="monotone" dataKey="pnl" stroke={color} strokeWidth={2} fill="url(#pnlGrad)" dot={false} />
      </AreaChart>
    </ResponsiveContainer>
  );
}