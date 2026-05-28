import React from 'react';
import { Label } from '@/components/ui/label';
import { Slider } from '@/components/ui/slider';
import { Switch } from '@/components/ui/switch';
import { Sparkles } from 'lucide-react';

export default function DynamicStrategyPanel({ form, setForm }) {
  const enabled = form.use_dynamic_strategy === true;

  return (
    <div className={`rounded-lg border p-4 space-y-4 transition-colors ${
      enabled ? 'border-profit/30 bg-profit/5' : 'border-border bg-muted/20'
    }`}>
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Sparkles className={`w-4 h-4 ${enabled ? 'text-profit' : 'text-muted-foreground'}`} />
          <div>
            <Label className="text-xs text-muted-foreground uppercase tracking-widest">Dynamic Strategy</Label>
            <p className="text-[11px] text-muted-foreground/70 mt-0.5">
              {enabled ? 'Using your tuned thresholds' : 'Using built-in core strategy defaults'}
            </p>
          </div>
        </div>
        <Switch
          checked={enabled}
          onCheckedChange={(v) => setForm(f => ({ ...f, use_dynamic_strategy: v }))}
        />
      </div>

      <div className={`grid grid-cols-2 gap-4 ${enabled ? '' : 'opacity-50 pointer-events-none'}`}>
        <div className="space-y-3">
          <Label className="text-xs text-muted-foreground uppercase tracking-widest">
            Min Confidence: <span className="text-profit font-mono">{(form.dynamic_min_confidence ?? 0.5).toFixed(2)}</span>
          </Label>
          <Slider
            min={0.30} max={0.90} step={0.05}
            value={[form.dynamic_min_confidence ?? 0.5]}
            onValueChange={([v]) => setForm(f => ({ ...f, dynamic_min_confidence: parseFloat(v.toFixed(2)) }))}
          />
        </div>
        <div className="space-y-3">
          <Label className="text-xs text-muted-foreground uppercase tracking-widest">
            Max Volatility: <span className="text-chart-3 font-mono">{((form.dynamic_volatility_threshold ?? 0.035) * 100).toFixed(2)}%</span>
          </Label>
          <Slider
            min={0.01} max={0.10} step={0.005}
            value={[form.dynamic_volatility_threshold ?? 0.035]}
            onValueChange={([v]) => setForm(f => ({ ...f, dynamic_volatility_threshold: parseFloat(v.toFixed(3)) }))}
          />
        </div>
        <div className="space-y-3">
          <Label className="text-xs text-muted-foreground uppercase tracking-widest">
            Max Spread: <span className="text-chart-3 font-mono">{((form.dynamic_spread_threshold ?? 0.005) * 100).toFixed(2)}%</span>
          </Label>
          <Slider
            min={0.001} max={0.02} step={0.001}
            value={[form.dynamic_spread_threshold ?? 0.005]}
            onValueChange={([v]) => setForm(f => ({ ...f, dynamic_spread_threshold: parseFloat(v.toFixed(3)) }))}
          />
        </div>
        <div className="space-y-3">
          <Label className="text-xs text-muted-foreground uppercase tracking-widest">
            Min OFI: <span className="text-chart-4 font-mono">{(form.dynamic_ofi_threshold ?? 0.10).toFixed(2)}</span>
          </Label>
          <Slider
            min={0.05} max={0.50} step={0.05}
            value={[form.dynamic_ofi_threshold ?? 0.10]}
            onValueChange={([v]) => setForm(f => ({ ...f, dynamic_ofi_threshold: parseFloat(v.toFixed(2)) }))}
          />
        </div>
      </div>
    </div>
  );
}