import React, { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Slider } from '@/components/ui/slider';
import { Badge } from '@/components/ui/badge';
import { base44 } from '@/api/base44Client';
import { Save } from 'lucide-react';
import DynamicStrategyPanel from './DynamicStrategyPanel';

export default function BotSettingsPanel({ config, onSaved }) {
  const [availableCoins, setAvailableCoins] = useState([]);
  const [form, setForm] = useState({
    wallet_address: config?.wallet_address || '0x3b3aad5cfaf13140883ab4d65c97421b5577bdc6',
    risk_per_trade: config?.risk_per_trade ?? 1,
    max_open_trades: config?.max_open_trades ?? 20,
    take_profit_pct: config?.take_profit_pct ?? 0.4,
    stop_loss_pct: config?.stop_loss_pct ?? 0.2,
    imbalance_threshold: config?.imbalance_threshold ?? 1.5,
    momentum_period: config?.momentum_period ?? 5,
    selected_coins: config?.selected_coins || ['SOL', 'AVAX', 'DOGE', 'ARB'],
    strategy_mode: config?.strategy_mode || 'relaxed',
    use_dynamic_strategy: config?.use_dynamic_strategy ?? false,
    dynamic_min_confidence: config?.dynamic_min_confidence ?? 0.5,
    dynamic_volatility_threshold: config?.dynamic_volatility_threshold ?? 0.035,
    dynamic_spread_threshold: config?.dynamic_spread_threshold ?? 0.005,
    dynamic_ofi_threshold: config?.dynamic_ofi_threshold ?? 0.10,
  });
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    base44.functions.invoke('hlPrice', {}).then(res => {
      const coins = res.data?.allCoins || [];
      setAvailableCoins(coins);
    }).catch(() => {});
  }, []);

  const toggleCoin = (coin) => {
    setForm(f => ({
      ...f,
      selected_coins: f.selected_coins.includes(coin)
        ? f.selected_coins.filter(c => c !== coin)
        : [...f.selected_coins, coin],
    }));
  };

  const save = async () => {
    setSaving(true);
    if (config?.id) {
      await base44.entities.BotConfig.update(config.id, form);
    } else {
      await base44.entities.BotConfig.create(form);
    }
    setSaving(false);
    onSaved?.();
  };

  return (
    <div className="space-y-6">
      <div className="space-y-2">
        <Label className="text-xs text-muted-foreground uppercase tracking-widest">Wallet Address</Label>
        <Input
          value={form.wallet_address}
          onChange={e => setForm(f => ({ ...f, wallet_address: e.target.value }))}
          className="font-mono text-xs bg-muted border-border"
          placeholder="0x..."
        />
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div className="space-y-3">
          <Label className="text-xs text-muted-foreground uppercase tracking-widest">Risk per Trade: <span className="text-profit font-mono">{form.risk_per_trade}%</span></Label>
          <Slider min={0.5} max={5} step={0.5} value={[form.risk_per_trade]} onValueChange={([v]) => setForm(f => ({ ...f, risk_per_trade: v }))} />
        </div>
        <div className="space-y-3">
          <Label className="text-xs text-muted-foreground uppercase tracking-widest">Max Open Trades: <span className="text-profit font-mono">{form.max_open_trades}</span></Label>
          <Slider min={1} max={20} step={1} value={[form.max_open_trades]} onValueChange={([v]) => setForm(f => ({ ...f, max_open_trades: v }))} />
        </div>
        <div className="space-y-3">
          <Label className="text-xs text-muted-foreground uppercase tracking-widest">Take Profit: <span className="text-profit font-mono">{form.take_profit_pct}%</span></Label>
          <Slider min={0.1} max={2} step={0.1} value={[form.take_profit_pct]} onValueChange={([v]) => setForm(f => ({ ...f, take_profit_pct: parseFloat(v.toFixed(1)) }))} />
        </div>
        <div className="space-y-3">
          <Label className="text-xs text-muted-foreground uppercase tracking-widest">Stop Loss: <span className="text-loss font-mono">{form.stop_loss_pct}%</span></Label>
          <Slider min={0.1} max={1} step={0.1} value={[form.stop_loss_pct]} onValueChange={([v]) => setForm(f => ({ ...f, stop_loss_pct: parseFloat(v.toFixed(1)) }))} />
        </div>
        <div className="space-y-3">
          <Label className="text-xs text-muted-foreground uppercase tracking-widest">Imbalance Threshold: <span className="text-chart-4 font-mono">{form.imbalance_threshold}x</span></Label>
          <Slider min={1.1} max={3} step={0.1} value={[form.imbalance_threshold]} onValueChange={([v]) => setForm(f => ({ ...f, imbalance_threshold: parseFloat(v.toFixed(1)) }))} />
        </div>
        <div className="space-y-3">
          <Label className="text-xs text-muted-foreground uppercase tracking-widest">Momentum Period: <span className="text-chart-4 font-mono">{form.momentum_period}</span></Label>
          <Slider min={3} max={20} step={1} value={[form.momentum_period]} onValueChange={([v]) => setForm(f => ({ ...f, momentum_period: v }))} />
        </div>
      </div>

      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <Label className="text-xs text-muted-foreground uppercase tracking-widest">
            Trading Pairs <span className="text-profit font-mono">({form.selected_coins.length}/{availableCoins.length})</span>
          </Label>
          <div className="flex gap-2">
            <button
              type="button"
              onClick={() => setForm(f => ({ ...f, selected_coins: [...availableCoins] }))}
              className="text-xs px-2 py-1 rounded border border-profit/30 text-profit hover:bg-profit/10 transition-colors"
            >
              Select All
            </button>
            <button
              type="button"
              onClick={() => setForm(f => ({ ...f, selected_coins: [] }))}
              className="text-xs px-2 py-1 rounded border border-border text-muted-foreground hover:bg-muted transition-colors"
            >
              Clear All
            </button>
          </div>
        </div>
        <div className="flex flex-wrap gap-2 max-h-48 overflow-y-auto">
          {availableCoins.length === 0 && <p className="text-xs text-muted-foreground">Loading pairs...</p>}
          {availableCoins.map(coin => (
            <Badge
              key={coin}
              variant="outline"
              className={`cursor-pointer transition-all font-mono text-xs py-1 px-3 ${
                form.selected_coins.includes(coin)
                  ? 'bg-profit/10 border-profit text-profit'
                  : 'border-border text-muted-foreground hover:border-muted-foreground'
              }`}
              onClick={() => toggleCoin(coin)}
            >
              {coin}
            </Badge>
          ))}
        </div>
      </div>

      <div className="rounded-lg border border-border p-4 space-y-3 bg-muted/20">
        <Label className="text-xs text-muted-foreground uppercase tracking-widest">Strategy Mode</Label>
        <div className="grid grid-cols-3 gap-2">
          <button
            type="button"
            onClick={() => setForm(f => ({ ...f, strategy_mode: 'relaxed' }))}
            className={`p-3 rounded-lg border text-left transition-all ${
              form.strategy_mode === 'relaxed'
                ? 'border-profit bg-profit/10 text-profit'
                : 'border-border bg-muted/30 text-muted-foreground hover:border-muted-foreground'
            }`}
          >
            <div className="text-xs font-semibold uppercase tracking-widest">Relaxed</div>
            <div className="text-[11px] mt-1 opacity-80">OFI + momentum. Fires often.</div>
          </button>
          <button
            type="button"
            onClick={() => setForm(f => ({ ...f, strategy_mode: 'moderate' }))}
            className={`p-3 rounded-lg border text-left transition-all ${
              form.strategy_mode === 'moderate'
                ? 'border-chart-4 bg-chart-4/10 text-chart-4'
                : 'border-border bg-muted/30 text-muted-foreground hover:border-muted-foreground'
            }`}
          >
            <div className="text-xs font-semibold uppercase tracking-widest">Moderate</div>
            <div className="text-[11px] mt-1 opacity-80">OFI + mom + sweep/absorb.</div>
          </button>
          <button
            type="button"
            onClick={() => setForm(f => ({ ...f, strategy_mode: 'strict' }))}
            className={`p-3 rounded-lg border text-left transition-all ${
              form.strategy_mode === 'strict'
                ? 'border-chart-3 bg-chart-3/10 text-chart-3'
                : 'border-border bg-muted/30 text-muted-foreground hover:border-muted-foreground'
            }`}
          >
            <div className="text-xs font-semibold uppercase tracking-widest">Strict</div>
            <div className="text-[11px] mt-1 opacity-80">Sweep + absorb + MSS.</div>
          </button>
        </div>
      </div>

      <DynamicStrategyPanel form={form} setForm={setForm} />

      <Button onClick={save} disabled={saving} className="w-full bg-primary text-primary-foreground font-semibold">
        <Save className="w-4 h-4 mr-2" />
        {saving ? 'Saving...' : 'Save Configuration'}
      </Button>
    </div>
  );
}