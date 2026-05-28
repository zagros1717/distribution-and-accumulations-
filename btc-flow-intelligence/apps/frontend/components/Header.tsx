"use client";

import { Activity, RefreshCw } from "lucide-react";
import type { Dashboard } from "@/lib/api";
import { verdictColor } from "@/lib/api";
import { fmtUsd, relTime } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";

interface Props {
  data: Dashboard | null;
  mockMode: boolean;
  onRefresh: () => void;
  refreshing: boolean;
}

export function Header({ data, mockMode, onRefresh, refreshing }: Props) {
  const color = data ? verdictColor(data.verdict) : "var(--muted)";
  const change = data?.btc_change_24h ?? null;
  const changeColor = change === null ? "var(--muted)" : change >= 0 ? "var(--accum)" : "var(--dist)";

  return (
    <header className="flex flex-col gap-4 border-b border-border pb-5 sm:flex-row sm:items-center sm:justify-between">
      <div className="flex items-center gap-4">
        <div className="flex h-10 w-10 items-center justify-center rounded-lg border border-border bg-panel">
          <Activity size={18} style={{ color }} />
        </div>
        <div>
          <h1 className="font-mono text-base font-bold tracking-tight">
            BTC<span className="text-muted">·</span>FLOW
            <span className="ml-1 text-[10px] font-normal uppercase tracking-[0.2em] text-muted">
              intelligence
            </span>
          </h1>
          <div className="flex items-center gap-2 text-[11px] text-muted">
            <span
              className="h-1.5 w-1.5 rounded-full animate-pulse-dot"
              style={{ background: color }}
            />
            {data ? data.verdict : "awaiting data"} ·{" "}
            {data ? `updated ${relTime(data.timestamp)}` : "—"}
          </div>
        </div>
      </div>

      <div className="flex items-center gap-5">
        <div className="text-right">
          <div className="font-mono text-xl font-bold tabular">{fmtUsd(data?.btc_price ?? null)}</div>
          <div className="font-mono text-xs tabular" style={{ color: changeColor }}>
            {change === null ? "—" : `${change >= 0 ? "▲" : "▼"} ${Math.abs(change).toFixed(2)}% 24h`}
          </div>
        </div>
        <Badge
          className="border-current"
          style={{ color: mockMode ? "var(--neutral)" : "var(--accum)" }}
          title={mockMode ? "Serving realistic mock data — add API keys to go live" : "Live data sources active"}
        >
          {mockMode ? "mock" : "live"}
        </Badge>
        <button
          onClick={onRefresh}
          disabled={refreshing}
          className="flex items-center gap-2 rounded-lg border border-border bg-panel px-3 py-2 text-xs text-text transition-colors hover:bg-panel-2 disabled:opacity-50"
        >
          <RefreshCw size={13} className={refreshing ? "animate-spin" : ""} />
          {refreshing ? "refreshing" : "refresh"}
        </button>
      </div>
    </header>
  );
}
