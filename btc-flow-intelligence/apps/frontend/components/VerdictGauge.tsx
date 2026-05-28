"use client";

import { motion } from "framer-motion";
import type { Confidence, Verdict } from "@/lib/api";
import { verdictColor } from "@/lib/api";
import { Card, CardContent } from "@/components/ui/card";

interface Props {
  score: number; // -2..+2
  verdict: Verdict;
  confidence: Confidence;
  dataQuality: number; // 0..1
}

const RADIUS = 120;
const STROKE = 16;
const CIRC = Math.PI * RADIUS; // semicircle length

// map score -2..+2 → 0..1 along the arc
function scoreToFraction(score: number): number {
  return Math.min(1, Math.max(0, (score + 2) / 4));
}

const CONF_DOTS: Record<Confidence, number> = { Low: 1, Medium: 2, High: 3 };

export function VerdictGauge({ score, verdict, confidence, dataQuality }: Props) {
  const frac = scoreToFraction(score);
  const color = verdictColor(verdict);
  const dash = CIRC * frac;

  return (
    <Card className="relative overflow-hidden" style={{ boxShadow: `0 0 60px -20px ${color}55` }}>
      <div
        className="pointer-events-none absolute inset-0 opacity-[0.12]"
        style={{ background: `radial-gradient(60% 80% at 50% 100%, ${color}, transparent)` }}
      />
      <CardContent className="relative flex flex-col items-center pt-8">
        <div className="text-[11px] font-semibold uppercase tracking-[0.22em] text-muted">
          24h Market Structure
        </div>

        <svg width="280" height="170" viewBox="0 0 280 170" className="mt-4">
          {/* track */}
          <path
            d="M 20 150 A 120 120 0 0 1 260 150"
            fill="none"
            stroke="var(--border)"
            strokeWidth={STROKE}
            strokeLinecap="round"
          />
          {/* tick zones (dist / neutral / accum) */}
          <path d="M 20 150 A 120 120 0 0 1 88 54" fill="none" stroke="var(--dist)" strokeOpacity="0.25" strokeWidth="3" />
          <path d="M 110 38 A 120 120 0 0 1 170 38" fill="none" stroke="var(--neutral)" strokeOpacity="0.3" strokeWidth="3" />
          <path d="M 192 54 A 120 120 0 0 1 260 150" fill="none" stroke="var(--accum)" strokeOpacity="0.25" strokeWidth="3" />
          {/* value arc */}
          <motion.path
            d="M 20 150 A 120 120 0 0 1 260 150"
            fill="none"
            stroke={color}
            strokeWidth={STROKE}
            strokeLinecap="round"
            strokeDasharray={CIRC}
            initial={{ strokeDashoffset: CIRC }}
            animate={{ strokeDashoffset: CIRC - dash }}
            transition={{ duration: 1.1, ease: [0.22, 1, 0.36, 1] }}
          />
          <text x="140" y="120" textAnchor="middle" className="fill-text font-mono" style={{ fontSize: 40, fontWeight: 700 }}>
            <tspan fill={color}>{score >= 0 ? "+" : ""}{score.toFixed(2)}</tspan>
          </text>
          <text x="140" y="145" textAnchor="middle" className="fill-muted font-mono" style={{ fontSize: 11 }}>
            weighted score
          </text>
          <text x="22" y="168" className="fill-muted font-mono" style={{ fontSize: 10 }}>−2</text>
          <text x="248" y="168" className="fill-muted font-mono" style={{ fontSize: 10 }}>+2</text>
        </svg>

        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
          className="mt-1 flex flex-col items-center"
        >
          <div className="font-mono text-3xl font-bold tracking-tight" style={{ color }}>
            {verdict}
          </div>
          <div className="mt-3 flex items-center gap-4 text-xs text-muted">
            <span className="flex items-center gap-1.5">
              confidence
              <span className="flex gap-1">
                {[1, 2, 3].map((n) => (
                  <span
                    key={n}
                    className="h-1.5 w-1.5 rounded-full"
                    style={{ background: n <= CONF_DOTS[confidence] ? color : "var(--border)" }}
                  />
                ))}
              </span>
              <span className="font-mono text-text">{confidence}</span>
            </span>
            <span className="text-border">|</span>
            <span>
              data quality <span className="font-mono text-text">{Math.round(dataQuality * 100)}%</span>
            </span>
          </div>
        </motion.div>
      </CardContent>
    </Card>
  );
}
