"use client";

import { motion } from "framer-motion";
import { CheckCircle2, CircleDashed } from "lucide-react";
import type { CategoryScore } from "@/lib/api";
import { scoreColor } from "@/lib/api";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

// score -2..+2 → 0..100% with 50% center
function offset(score: number): number {
  return ((score + 2) / 4) * 100;
}

export function SignalMatrix({ categories }: { categories: CategoryScore[] }) {
  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between">
        <CardTitle>Signal Matrix · Weighted Scoring</CardTitle>
        <span className="font-mono text-[10px] text-muted">−2 dist · 0 neutral · +2 accum</span>
      </CardHeader>
      <CardContent className="space-y-3.5">
        {categories.map((c, i) => {
          const color = scoreColor(c.score);
          return (
            <motion.div
              key={c.category}
              initial={{ opacity: 0, x: -8 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: i * 0.05 }}
              className="grid grid-cols-[1fr_auto] items-center gap-x-3"
            >
              <div className="flex items-center gap-2">
                <span className="text-sm text-text">{c.label}</span>
                <span className="font-mono text-[10px] text-muted">{Math.round(c.weight * 100)}%</span>
                {c.is_live ? (
                  <CheckCircle2 size={12} className="text-accum" />
                ) : (
                  <CircleDashed size={12} className="text-muted" />
                )}
              </div>
              <div className="flex items-center gap-3">
                <span className="w-12 text-right font-mono text-sm tabular" style={{ color }}>
                  {c.score >= 0 ? "+" : ""}
                  {c.score.toFixed(2)}
                </span>
                <span className="w-14 text-right font-mono text-[11px] text-muted tabular">
                  {c.weighted >= 0 ? "+" : ""}
                  {c.weighted.toFixed(3)}
                </span>
              </div>

              {/* diverging bar */}
              <div className="col-span-2 relative h-2 rounded-full bg-panel-2">
                <div className="absolute left-1/2 top-0 h-full w-px bg-border" />
                <motion.div
                  className="absolute top-0 h-full rounded-full"
                  style={{
                    background: color,
                    ...(c.score >= 0
                      ? { left: "50%", transformOrigin: "left" }
                      : { right: "50%", transformOrigin: "right" }),
                  }}
                  initial={{ width: 0 }}
                  animate={{ width: `${Math.abs(offset(c.score) - 50)}%` }}
                  transition={{ delay: i * 0.05 + 0.15, duration: 0.6, ease: "easeOut" }}
                />
              </div>
            </motion.div>
          );
        })}
      </CardContent>
    </Card>
  );
}
