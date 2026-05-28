// Mirror of packages/shared/scoring_spec.py — keep in sync.
export const ACCUMULATION_THRESHOLD = 0.5;
export const DISTRIBUTION_THRESHOLD = -0.5;

export type Verdict = "Accumulation" | "Distribution" | "Mixed/Neutral";
export type Confidence = "Low" | "Medium" | "High";

export interface CategoryScore {
  category: string;
  label: string;
  weight: number;
  score: number;
  weighted: number;
  is_live: boolean;
  signal_count: number;
}

export interface Dashboard {
  snapshot_id: number;
  timestamp: string;
  btc_price: number | null;
  btc_change_24h: number | null;
  final_score: number;
  verdict: Verdict;
  confidence: Confidence;
  data_quality: number;
  categories: CategoryScore[];
}

export interface SignalReading {
  category: string;
  source: string;
  metric: string;
  value: number | null;
  change_24h: number | null;
  score: number;
  is_live: boolean;
  raw: Record<string, unknown> | null;
}

export interface HistoryPoint {
  snapshot_id: number;
  timestamp: string;
  btc_price: number | null;
  final_score: number;
  verdict: Verdict;
  confidence: Confidence;
}

export interface HealthInfo {
  status: string;
  environment: string;
  mock_mode: boolean;
  sources: Record<string, "live" | "mock">;
  last_snapshot: string | null;
}

const BASE = process.env.NEXT_PUBLIC_API_BASE ?? "";

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`, { cache: "no-store" });
  if (!res.ok) throw new Error(`${path} → ${res.status}`);
  return res.json() as Promise<T>;
}

export const api = {
  dashboard: () => get<Dashboard>("/api/dashboard"),
  signals: () => get<{ signals: SignalReading[] }>("/api/signals"),
  history: (limit = 168) => get<HistoryPoint[]>(`/api/history?limit=${limit}`),
  report: () => get<{ markdown: string }>("/api/report/latest"),
  health: () => get<HealthInfo>("/api/health"),
  refresh: async (): Promise<Dashboard> => {
    const res = await fetch(`${BASE}/api/refresh`, { method: "POST" });
    if (!res.ok) throw new Error(`refresh → ${res.status}`);
    return res.json();
  },
};

export function verdictColor(v: Verdict): string {
  if (v === "Accumulation") return "var(--accum)";
  if (v === "Distribution") return "var(--dist)";
  return "var(--neutral)";
}

export function scoreColor(score: number): string {
  if (score > 0.05) return "var(--accum)";
  if (score < -0.05) return "var(--dist)";
  return "var(--neutral)";
}
