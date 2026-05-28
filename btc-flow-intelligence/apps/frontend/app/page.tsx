"use client";

import { useCallback, useEffect, useState } from "react";
import { motion } from "framer-motion";
import {
  api,
  type Dashboard,
  type HistoryPoint,
  type SignalReading,
} from "@/lib/api";
import { Header } from "@/components/Header";
import { VerdictGauge } from "@/components/VerdictGauge";
import { SignalMatrix } from "@/components/SignalMatrix";
import { HistoryChart } from "@/components/HistoryChart";
import { SignalsTable } from "@/components/SignalsTable";
import { ReportPanel } from "@/components/ReportPanel";

function Skeleton({ className = "" }: { className?: string }) {
  return <div className={`animate-pulse rounded-xl border border-border bg-panel/60 ${className}`} />;
}

export default function Page() {
  const [dash, setDash] = useState<Dashboard | null>(null);
  const [signals, setSignals] = useState<SignalReading[]>([]);
  const [history, setHistory] = useState<HistoryPoint[]>([]);
  const [report, setReport] = useState<string>("");
  const [mockMode, setMockMode] = useState(true);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      setError(null);
      let dashboardPromise = api.dashboard();
      try {
        await dashboardPromise;
      } catch (err) {
        // First deploys may have RUN_ON_STARTUP=false or an empty DB.
        // Bootstrap one snapshot automatically instead of leaving the UI blank.
        if ((err as Error).message.includes("404")) {
          dashboardPromise = api.refresh();
        }
      }
      const [d, s, h, r, health] = await Promise.all([
        dashboardPromise,
        api.signals(),
        api.history(168),
        api.report().catch(() => ({ markdown: "" })),
        api.health().catch(() => null),
      ]);
      setDash(d);
      setSignals(s.signals);
      setHistory(h);
      setReport(r.markdown);
      if (health) setMockMode(health.mock_mode);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
    const id = setInterval(load, 60_000); // poll every minute
    return () => clearInterval(id);
  }, [load]);

  const refresh = async () => {
    setRefreshing(true);
    try {
      await api.refresh();
      await load();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setRefreshing(false);
    }
  };

  return (
    <main className="mx-auto max-w-6xl px-4 py-6 sm:px-6 sm:py-8">
      <Header data={dash} mockMode={mockMode} onRefresh={refresh} refreshing={refreshing} />

      {error && (
        <div className="mt-4 rounded-lg border border-dist/40 bg-dist/10 px-4 py-3 text-sm text-dist">
          Could not reach the API ({error}). Is the backend running on the configured BACKEND_URL?
        </div>
      )}

      {loading ? (
        <div className="mt-6 grid gap-4 lg:grid-cols-5">
          <Skeleton className="h-[360px] lg:col-span-2" />
          <Skeleton className="h-[360px] lg:col-span-3" />
          <Skeleton className="h-[240px] lg:col-span-3" />
          <Skeleton className="h-[240px] lg:col-span-2" />
        </div>
      ) : (
        <motion.div
          initial="hidden"
          animate="show"
          variants={{ show: { transition: { staggerChildren: 0.08 } } }}
          className="mt-6 grid gap-4 lg:grid-cols-5"
        >
          {dash && (
            <>
              <motion.div variants={fade} className="lg:col-span-2">
                <VerdictGauge
                  score={dash.final_score}
                  verdict={dash.verdict}
                  confidence={dash.confidence}
                  dataQuality={dash.data_quality}
                />
              </motion.div>
              <motion.div variants={fade} className="lg:col-span-3">
                <SignalMatrix categories={dash.categories} />
              </motion.div>
            </>
          )}
          <motion.div variants={fade} className="lg:col-span-3">
            <HistoryChart data={history} />
          </motion.div>
          <motion.div variants={fade} className="lg:col-span-2">
            {report ? <ReportPanel markdown={report} /> : <Skeleton className="h-[240px]" />}
          </motion.div>
          <motion.div variants={fade} className="lg:col-span-5">
            <SignalsTable signals={signals} />
          </motion.div>
        </motion.div>
      )}

      <footer className="mt-8 border-t border-border pt-4 text-center font-mono text-[10px] text-muted">
        BTC FLOW INTELLIGENCE · automated market-structure analysis, not financial advice ·
        {mockMode ? " running in mock mode — add API keys to go live" : " live data active"}
      </footer>
    </main>
  );
}

const fade = {
  hidden: { opacity: 0, y: 12 },
  show: { opacity: 1, y: 0 },
};
