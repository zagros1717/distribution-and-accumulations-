"use client";

import {
  Area,
  AreaChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { HistoryPoint } from "@/lib/api";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

export function HistoryChart({ data }: { data: HistoryPoint[] }) {
  const points = data.map((d) => ({
    t: new Date(d.timestamp).toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit" }),
    score: d.final_score,
    verdict: d.verdict,
  }));

  return (
    <Card>
      <CardHeader>
        <CardTitle>Historical Score · {data.length} snapshots</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="h-[200px] w-full">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={points} margin={{ top: 8, right: 8, left: -24, bottom: 0 }}>
              <defs>
                <linearGradient id="scoreFill" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="var(--accum)" stopOpacity={0.35} />
                  <stop offset="50%" stopColor="var(--neutral)" stopOpacity={0.12} />
                  <stop offset="100%" stopColor="var(--dist)" stopOpacity={0.25} />
                </linearGradient>
              </defs>
              <XAxis dataKey="t" tick={{ fill: "var(--muted)", fontSize: 10 }} tickLine={false} axisLine={{ stroke: "var(--border)" }} minTickGap={32} />
              <YAxis domain={[-2, 2]} ticks={[-2, -1, 0, 1, 2]} tick={{ fill: "var(--muted)", fontSize: 10 }} tickLine={false} axisLine={false} />
              <ReferenceLine y={0.5} stroke="var(--accum)" strokeDasharray="3 3" strokeOpacity={0.4} />
              <ReferenceLine y={-0.5} stroke="var(--dist)" strokeDasharray="3 3" strokeOpacity={0.4} />
              <ReferenceLine y={0} stroke="var(--border)" />
              <Tooltip
                contentStyle={{
                  background: "var(--panel-2)",
                  border: "1px solid var(--border)",
                  borderRadius: 8,
                  fontSize: 12,
                }}
                labelStyle={{ color: "var(--muted)" }}
                formatter={(v: number, _n, p) => [`${v >= 0 ? "+" : ""}${v.toFixed(2)} · ${p.payload.verdict}`, "score"]}
              />
              <Area type="monotone" dataKey="score" stroke="var(--text)" strokeWidth={1.5} fill="url(#scoreFill)" />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  );
}
