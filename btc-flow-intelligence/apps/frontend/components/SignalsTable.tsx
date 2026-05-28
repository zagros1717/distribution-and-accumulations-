"use client";

import { useMemo, useState } from "react";
import type { SignalReading } from "@/lib/api";
import { scoreColor } from "@/lib/api";
import { fmt } from "@/lib/utils";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

const SCORE_LABEL: Record<number, string> = {
  2: "++", 1: "+", 0: "0", "-1": "−", "-2": "−−",
};

export function SignalsTable({ signals }: { signals: SignalReading[] }) {
  const categories = useMemo(
    () => ["all", ...Array.from(new Set(signals.map((s) => s.category)))],
    [signals]
  );
  const [filter, setFilter] = useState("all");
  const rows = signals.filter((s) => filter === "all" || s.category === filter);

  return (
    <Card>
      <CardHeader className="flex flex-row flex-wrap items-center justify-between gap-2">
        <CardTitle>Confirmed 24h Signals</CardTitle>
        <div className="flex flex-wrap gap-1">
          {categories.map((c) => (
            <button
              key={c}
              onClick={() => setFilter(c)}
              className={`rounded-md border px-2 py-0.5 font-mono text-[10px] uppercase tracking-wide transition-colors ${
                filter === c
                  ? "border-text/40 bg-panel-2 text-text"
                  : "border-border text-muted hover:text-text"
              }`}
            >
              {c.replace(/_/g, " ")}
            </button>
          ))}
        </div>
      </CardHeader>
      <CardContent>
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="text-[10px] uppercase tracking-wider text-muted">
              <tr className="border-b border-border">
                <th className="pb-2 pr-3 font-medium">Source</th>
                <th className="pb-2 pr-3 font-medium">Metric</th>
                <th className="pb-2 pr-3 text-right font-medium">Value</th>
                <th className="pb-2 pr-3 text-right font-medium">Δ24h</th>
                <th className="pb-2 pr-3 text-center font-medium">Score</th>
                <th className="pb-2 text-center font-medium">Src</th>
              </tr>
            </thead>
            <tbody className="font-mono">
              {rows.map((s, i) => (
                <tr key={i} className="border-b border-border/50 last:border-0">
                  <td className="py-1.5 pr-3 text-muted">{s.source}</td>
                  <td className="py-1.5 pr-3 text-text">{s.metric.replace(/_/g, " ")}</td>
                  <td className="py-1.5 pr-3 text-right tabular">{fmt(s.value)}</td>
                  <td className="py-1.5 pr-3 text-right tabular text-muted">{s.change_24h === null ? "—" : fmt(s.change_24h)}</td>
                  <td className="py-1.5 pr-3 text-center font-bold" style={{ color: scoreColor(s.score) }}>
                    {SCORE_LABEL[s.score] ?? s.score}
                  </td>
                  <td className="py-1.5 text-center">
                    <span
                      className="inline-block h-1.5 w-1.5 rounded-full"
                      style={{ background: s.is_live ? "var(--accum)" : "var(--neutral)" }}
                      title={s.is_live ? "live" : "mock"}
                    />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </CardContent>
    </Card>
  );
}
