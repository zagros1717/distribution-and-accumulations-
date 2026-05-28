"""
btcalpha.live.engine
~~~~~~~~~~~~~~~~~~~~
Pipeline orchestrator for BTC Swing Alpha.

Safety rule:
  Live comparison/verdict must never be computed from the full backtest curve,
  because the full curve includes the memorized train segment. User-facing
  comparison is scoped to the model's independent test window only. If no enough
  test-only similar trades exist, the verdict is intentionally cautious.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from btcalpha.config import get_config, get_logger
from btcalpha.data.exchanges import fetch_latest_price
from btcalpha.data.merge import build_dataset
from btcalpha.features.engineering import build_features
from btcalpha.regime.detector import detect_regime
from btcalpha.model.alpha_model import AlphaModel, get_or_train_model
from btcalpha.strategy.signal import Strategy
from btcalpha.backtest.engine import Backtester, BacktestResult

log = get_logger("live")


@dataclass
class PipelineSnapshot:
    timeframe: str
    generated_at: str
    dataset: pd.DataFrame
    features: dict
    regime: pd.DataFrame
    model: AlphaModel
    proba: pd.DataFrame
    decisions: pd.DataFrame
    backtest: BacktestResult
    live: dict


def run_pipeline(timeframe: str, force_retrain: bool = False) -> PipelineSnapshot:
    log.info("=" * 50)
    log.info("اجرای زنجیره برای تایم‌فریم: %s", timeframe)
    log.info("=" * 50)

    dataset = build_dataset(timeframe)
    regime = detect_regime(dataset)
    dataset = dataset.join(regime[["regime_score"]])
    feat = build_features(dataset, timeframe)
    model = get_or_train_model(feat, force_retrain=force_retrain)
    proba = model.predict_proba(feat["X"])

    strat = Strategy(timeframe=timeframe)
    regime_aligned = regime.reindex(proba.index).ffill()
    decisions = strat.decide_series(proba, regime_aligned, feat["raw"])

    bt = Backtester().run(decisions, feat["raw"])
    live = _live_decision_from_snapshot(decisions, proba, regime_aligned, bt, feat, timeframe, model)

    return PipelineSnapshot(
        timeframe=timeframe,
        generated_at=datetime.now(timezone.utc).isoformat(),
        dataset=dataset,
        features=feat,
        regime=regime,
        model=model,
        proba=proba,
        decisions=decisions,
        backtest=bt,
        live=live,
    )


def _bias_from_proba(last_proba: dict, decision: dict) -> dict:
    p_up = float(last_proba.get("p_up", 0.0))
    p_down = float(last_proba.get("p_down", 0.0))
    p_neutral = float(last_proba.get("p_neutral", 0.0))
    raw_alpha = p_up - p_down
    if abs(raw_alpha) < 0.05 or max(p_up, p_down) <= p_neutral:
        bias = "neutral"
    elif raw_alpha > 0:
        bias = "bullish"
    else:
        bias = "bearish"
    return {
        "bias": bias,
        "raw_alpha": raw_alpha,
        "directional_confidence": max(p_up, p_down),
        "neutral_probability": p_neutral,
        "trade_gate_direction": decision.get("direction"),
        "trade_gate_blocked": decision.get("direction") == "flat" and bias != "neutral",
        "block_reasons": decision.get("reasons", [])[3:] if decision.get("direction") == "flat" else [],
    }


def _test_window(model: AlphaModel) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    meta = model.meta
    if meta is None or not meta.test_start or not meta.test_end:
        return None, None
    try:
        return pd.Timestamp(meta.test_start), pd.Timestamp(meta.test_end)
    except Exception:  # noqa: BLE001
        return None, None


def _trades_in_test_window(bt: BacktestResult, model: AlphaModel) -> list:
    start, end = _test_window(model)
    if start is None or end is None:
        return []
    out = []
    for t in bt.trades:
        try:
            entry_ts = pd.Timestamp(t.entry_time)
        except Exception:  # noqa: BLE001
            continue
        if start <= entry_ts <= end:
            out.append(t)
    return out


def _win_rate(trades: list) -> float | None:
    if not trades:
        return None
    return sum(1 for t in trades if t.pnl > 0) / len(trades) * 100.0


def _avg_pct(trades: list) -> float | None:
    if not trades:
        return None
    return float(np.mean([t.pnl_pct for t in trades]))


def _live_decision_from_snapshot(decisions, proba, regime, bt, feat, timeframe: str, model: AlphaModel) -> dict:
    last_ts = decisions.index[-1]
    last = decisions.loc[last_ts].to_dict()
    last_proba = proba.loc[last_ts].to_dict()

    cfg = get_config()
    primary_exchange = cfg.get("data", {}).get("primary_exchange", "bitfinex")
    market_price = fetch_latest_price(primary_exchange)
    candle_price = float(feat["raw"].loc[last_ts, "close"])
    if market_price.get("price") is not None:
        last["market_price"] = float(market_price["price"])
        last["candle_close_price"] = candle_price
        last["price_source"] = "ticker"
        last["price_delta_vs_candle_pct"] = (float(market_price["price"]) / candle_price - 1.0) * 100.0 if candle_price else None
    else:
        last["market_price"] = None
        last["candle_close_price"] = candle_price
        last["price_source"] = "candle_close"
        last["price_delta_vs_candle_pct"] = None

    test_trades = _trades_in_test_window(bt, model)
    matching_trades = [t for t in test_trades if t.direction == last["direction"]]
    hist_win_rate = _win_rate(matching_trades)
    hist_avg_pct = _avg_pct(matching_trades)

    regime_trades = []
    for t in test_trades:
        try:
            entry_ts = pd.Timestamp(t.entry_time)
            if entry_ts in regime.index and regime.at[entry_ts, "regime_label"] == last["regime"]:
                regime_trades.append(t)
        except Exception:
            pass
    regime_win_rate = _win_rate(regime_trades)
    test_win_rate = _win_rate(test_trades)
    test_avg_pct = _avg_pct(test_trades)

    start, end = _test_window(model)
    comparison = {
        "scope": "test_only",
        "test_start": str(start) if start is not None else None,
        "test_end": str(end) if end is not None else None,
        "historical_win_rate_same_direction": round(hist_win_rate, 1) if hist_win_rate is not None else None,
        "historical_avg_pct_same_direction": round(hist_avg_pct, 3) if hist_avg_pct is not None else None,
        "historical_win_rate_same_regime": round(regime_win_rate, 1) if regime_win_rate is not None else None,
        "n_similar_trades": len(matching_trades),
        "test_only_n_trades": len(test_trades),
        "test_only_win_rate": round(test_win_rate, 1) if test_win_rate is not None else None,
        "test_only_avg_trade_pct": round(test_avg_pct, 3) if test_avg_pct is not None else None,
        "full_backtest_overall_win_rate_hidden_from_verdict": bt.metrics["win_rate"],
        "full_backtest_sharpe_hidden_from_verdict": bt.metrics["sharpe"],
    }

    verdict = _build_verdict(last, comparison)
    return {
        "timestamp": str(last_ts),
        "timeframe": timeframe,
        "decision": last,
        "proba": last_proba,
        "model_bias": _bias_from_proba(last_proba, last),
        "market": market_price,
        "comparison": comparison,
        "verdict": verdict,
    }


def _build_verdict(decision: dict, comparison: dict) -> dict:
    if decision["direction"] == "flat":
        return {"level": "neutral", "text": "سیستم در حال حاضر سیگنال معاملاتی ندارد — بهتر است بیرون بمانی."}

    if comparison.get("scope") != "test_only":
        return {
            "level": "weak",
            "text": "مقایسه‌ی مستقل تست موجود نیست؛ سیگنال فقط bias است و قابل اتکا نیست.",
            "notes": ["comparison scope is not test_only"],
            "score": -2,
        }

    hist_wr = comparison["historical_win_rate_same_direction"]
    regime_wr = comparison["historical_win_rate_same_regime"]
    n_similar = int(comparison.get("n_similar_trades") or 0)
    test_n = int(comparison.get("test_only_n_trades") or 0)
    test_avg = comparison.get("test_only_avg_trade_pct")
    score = 0
    notes = ["مقایسه فقط بر اساس دوره‌ی test مستقل انجام شده است"]

    if test_n < 20:
        score -= 2
        notes.append("تعداد معاملات test برای verdict کافی نیست")

    if n_similar < 5:
        score -= 1
        notes.append("نمونه‌ی هم‌جهت در test کم است")

    if decision["confidence"] >= 0.65:
        score += 1
        notes.append("اطمینان مدل بالا، اما خام و کالیبره‌نشده")
    elif decision["confidence"] < 0.5:
        score -= 1
        notes.append("اطمینان مدل پایین")

    if hist_wr is not None:
        if hist_wr >= 55 and n_similar >= 10:
            score += 1
            notes.append(f"معاملات هم‌جهت در test قابل قبول بوده ({hist_wr:.0f}٪)")
        elif hist_wr < 50:
            score -= 1
            notes.append(f"معاملات هم‌جهت در test ضعیف بوده ({hist_wr:.0f}٪)")

    if regime_wr is not None:
        if regime_wr >= 55 and len(notes) < 8:
            score += 1
            notes.append(f"رژیم مشابه در test بهتر بوده ({regime_wr:.0f}٪)")
        elif regime_wr < 50:
            score -= 1
            notes.append(f"رژیم مشابه در test ضعیف بوده ({regime_wr:.0f}٪)")

    if test_avg is not None and test_avg <= 0:
        score -= 1
        notes.append(f"میانگین سود معامله در test مثبت نیست ({test_avg:+.3f}٪)")

    if score >= 2:
        level, text = "moderate", "سیگنال با بخشی از شواهد test هم‌خوان است، اما همچنان نیازمند Trust Gate است."
    elif score >= 0:
        level, text = "weak", "سیگنال خام ML ضعیف/متوسط است و بدون تایید Trust Gate قابل معامله نیست."
    else:
        level, text = "weak", "سیگنال خام ML با شواهد test مستقل تایید نمی‌شود؛ قابل معامله نیست."

    return {"level": level, "text": text, "notes": notes, "score": score}


def live_decision(timeframe: str) -> dict:
    dataset = build_dataset(timeframe)
    regime = detect_regime(dataset)
    dataset = dataset.join(regime[["regime_score"]])
    feat = build_features(dataset, timeframe)

    # Use the same schema-checked loader as the full pipeline. This prevents
    # /api/live from bypassing stale-model protection after feature schema
    # changes such as removing macro_* columns from model inputs.
    model = get_or_train_model(feat, force_retrain=False)

    proba = model.predict_proba(feat["X"])
    strat = Strategy(timeframe=timeframe)
    regime_aligned = regime.reindex(proba.index).ffill()
    decisions = strat.decide_series(proba.tail(1), regime_aligned, feat["raw"])

    last_ts = decisions.index[-1]
    decision = decisions.loc[last_ts].to_dict()
    market_price = fetch_latest_price(get_config().get("data", {}).get("primary_exchange", "bitfinex"))
    candle_price = float(feat["raw"].loc[last_ts, "close"])
    if market_price.get("price") is not None:
        decision["market_price"] = float(market_price["price"])
        decision["candle_close_price"] = candle_price
        decision["price_source"] = "ticker"
        decision["price_delta_vs_candle_pct"] = (float(market_price["price"]) / candle_price - 1.0) * 100.0 if candle_price else None

    last_proba = proba.loc[last_ts].to_dict()
    return {
        "timeframe": timeframe,
        "timestamp": str(last_ts),
        "decision": decision,
        "proba": last_proba,
        "model_bias": _bias_from_proba(last_proba, decision),
        "market": market_price,
        "regime": regime_aligned.loc[last_ts].to_dict(),
        "model_trained_at": model.meta.trained_at if model.meta else None,
    }
