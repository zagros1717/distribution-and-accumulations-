"""
btcalpha.api.server
~~~~~~~~~~~~~~~~~~~
FastAPI backend for BTC Swing Alpha.

All JSON endpoints use SafeJSONResponse explicitly so NaN/Inf values from
numpy/pandas/model metrics never crash Starlette's strict JSON renderer.
"""
from __future__ import annotations

import copy
import json
import math
import os
import time
import traceback
from dataclasses import asdict
from numbers import Real
from pathlib import Path
from typing import Any, Dict

from btcalpha.api.evaluation import (
    backtest_segments,
    model_quality_audit,
    side_audit,
    threshold_sweep_audit,
    trust_gate,
    walk_forward_audit,
)
from btcalpha.api.long_edge_benchmark import long_edge_benchmark_audit
from btcalpha.api.walkforward_sides import walk_forward_side_audit
from btcalpha.config import CONFIG_PATH, PROJECT_ROOT, get_config, get_logger
from btcalpha.data.exchanges import fetch_latest_price
from btcalpha.live.collector import collect_once, get_collector_status, start_background_collector
from btcalpha.live.engine import PipelineSnapshot, live_decision, run_pipeline

log = get_logger("api")

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse, JSONResponse
except ImportError:  # noqa: BLE001
    raise SystemExit("FastAPI نصب نیست. اجرا کن: pip install fastapi uvicorn")


def _clean_json(obj: Any) -> Any:
    """Recursively convert NaN/Inf/numpy/pandas-ish values to JSON-safe data."""
    if obj is None or isinstance(obj, (str, bool)):
        return obj

    if isinstance(obj, Real):
        val = float(obj)
        if not math.isfinite(val):
            return None
        if hasattr(obj, "item"):
            try:
                return obj.item()
            except Exception:  # noqa: BLE001
                return val
        return obj

    if isinstance(obj, dict):
        return {str(k): _clean_json(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple, set)):
        return [_clean_json(v) for v in obj]

    if hasattr(obj, "tolist"):
        try:
            return _clean_json(obj.tolist())
        except Exception:  # noqa: BLE001
            pass

    if hasattr(obj, "item"):
        try:
            return _clean_json(obj.item())
        except Exception:  # noqa: BLE001
            pass

    if hasattr(obj, "isoformat"):
        try:
            return obj.isoformat()
        except Exception:  # noqa: BLE001
            return str(obj)

    try:
        json.dumps(obj, allow_nan=False)
        return obj
    except Exception:  # noqa: BLE001
        return None if str(obj) in {"nan", "NaN", "<NA>", "NaT"} else str(obj)


class SafeJSONResponse(JSONResponse):
    def render(self, content: Any) -> bytes:
        return json.dumps(
            _clean_json(content),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")


def _json(content: Any, status_code: int = 200) -> SafeJSONResponse:
    return SafeJSONResponse(content=content, status_code=status_code)


app = FastAPI(title="BTC Swing Alpha API", version="1.18", default_response_class=SafeJSONResponse)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_CACHE: Dict[str, dict] = {}
FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"
FRONTEND_FILE = FRONTEND_DIR / "dashboard.html"
FALLBACK_FRONTEND_FILE = FRONTEND_DIR / "index.html"


def _update_cache_from_collector(tf: str, snap: PipelineSnapshot) -> None:
    _CACHE[tf] = {"t": time.time(), "snap": snap, "source": "collector"}


@app.on_event("startup")
async def startup_event():
    start_background_collector(on_snapshot=_update_cache_from_collector)


def _serve_dashboard():
    path = FRONTEND_FILE if FRONTEND_FILE.exists() else FALLBACK_FRONTEND_FILE
    if not path.exists():
        raise HTTPException(404, "frontend dashboard file not found")
    return FileResponse(path)


def _valid_tf(tf: str) -> str:
    cfg = get_config()
    if tf not in cfg["data"]["timeframes"]:
        raise HTTPException(404, f"تایم‌فریم نامعتبر: {tf}")
    return tf


def _get_snapshot(tf: str, force: bool = False) -> PipelineSnapshot:
    cfg = get_config()
    ttl = cfg["api"]["refresh_seconds"]
    now = time.time()
    cached = _CACHE.get(tf)
    if cached and not force and (now - cached["t"]) < ttl:
        return cached["snap"]

    log.info("اجرای زنجیره برای %s (کش منقضی/خالی)", tf)
    snap = run_pipeline(tf, force_retrain=force)
    _CACHE[tf] = {"t": now, "snap": snap, "source": "api"}
    return snap


def _meta_dict(model) -> dict:
    if not model.meta:
        return {}
    try:
        return asdict(model.meta)
    except TypeError:
        return dict(model.meta.__dict__)


def _runtime_dict() -> dict:
    cfg = get_config()
    return {
        "api_version": app.version,
        "cwd": os.getcwd(),
        "project_root": str(PROJECT_ROOT),
        "config_path": str(CONFIG_PATH),
        "cache_dir": cfg["data"].get("cache_dir"),
        "model_dir": cfg["model"].get("model_dir"),
        "collector_dir": cfg.get("collector", {}).get("save_dir"),
        "timeframes": cfg["data"].get("timeframes"),
        "railway_git_commit_sha": os.getenv("RAILWAY_GIT_COMMIT_SHA"),
        "railway_deployment_id": os.getenv("RAILWAY_DEPLOYMENT_ID"),
        "railway_environment": os.getenv("RAILWAY_ENVIRONMENT_NAME"),
        "railway_service": os.getenv("RAILWAY_SERVICE_NAME"),
        "config_mtime": CONFIG_PATH.stat().st_mtime if CONFIG_PATH.exists() else None,
    }


def _current_market_dict() -> dict:
    cfg = get_config()
    primary = cfg.get("data", {}).get("primary_exchange", "bitfinex")
    return fetch_latest_price(primary)


def _display_prices(snap: PipelineSnapshot) -> dict:
    candle_last_price = float(snap.dataset["close"].iloc[-1]) if len(snap.dataset) else None
    market = snap.live.get("market") if isinstance(snap.live, dict) else None
    if not isinstance(market, dict):
        market = {}
    market_price = market.get("price")
    if market_price is None and isinstance(snap.live.get("decision"), dict):
        market_price = snap.live["decision"].get("market_price")
    if market_price is None:
        try:
            fresh_market = _current_market_dict()
            market_price = fresh_market.get("price")
            market = fresh_market
        except Exception:  # noqa: BLE001
            pass
    display_price = float(market_price) if market_price is not None else candle_last_price
    return {
        "last_price": display_price,
        "current_price": display_price,
        "market_price": float(market_price) if market_price is not None else None,
        "candle_last_price": candle_last_price,
        "price_source": "ticker" if market_price is not None else "candle_close",
        "price_delta_vs_candle_pct": (display_price / candle_last_price - 1.0) * 100.0 if display_price is not None and candle_last_price else None,
        "market": market,
    }


def _apply_trust_gate_to_live(live: dict, gate: dict) -> dict:
    """
    Research-first behavior:
    Trust Gate is advisory only. It must never rewrite raw alpha direction,
    position, probabilities, bias, stop loss, or take profit.
    """
    out = copy.deepcopy(live) if isinstance(live, dict) else {}

    gate_passed = bool(gate.get("enabled", False))
    reasons = gate.get("reasons") or []
    warnings = gate.get("warnings") or []

    out["trust_gate_advisory"] = gate
    out["trust_gate_blocked_live"] = False
    out["alpha_status"] = {
        "status": "alpha_candidate" if gate_passed else "no_confirmed_alpha",
        "research_only": True,
        "raw_signal_is_not_blocked": True,
        "gate_passed": gate_passed,
        "notes": reasons + warnings,
    }

    return out

    reasons = gate.get("reasons") or ["timeframe disabled by trust gate"]
    decision = out.get("decision") if isinstance(out.get("decision"), dict) else {}
    original_direction = decision.get("direction")
    original_position = decision.get("position")
    decision["direction"] = "flat"
    decision["position"] = 0.0
    decision["stop_loss"] = None
    decision["take_profit"] = None
    decision["trust_gate_blocked"] = True
    decision["trust_gate_original_direction"] = original_direction
    decision["trust_gate_original_position"] = original_position
    decision["trust_gate_reasons"] = reasons
    base_reasons = decision.get("reasons") if isinstance(decision.get("reasons"), list) else []
    decision["reasons"] = list(base_reasons) + ["Trust Gate: تایم‌فریم در تست مستقل edge تاییدشده ندارد — سیگنال اجرایی مسدود شد"]
    out["decision"] = decision

    model_bias = out.get("model_bias") if isinstance(out.get("model_bias"), dict) else {}
    model_bias["trade_gate_blocked"] = True
    model_bias["trust_gate_blocked"] = True
    model_bias["trade_gate_direction"] = "flat"
    block_reasons = model_bias.get("block_reasons") if isinstance(model_bias.get("block_reasons"), list) else []
    model_bias["block_reasons"] = list(block_reasons) + reasons
    out["model_bias"] = model_bias

    out["verdict"] = {
        "level": "neutral",
        "text": "Trust Gate این تایم‌فریم را تایید نکرده؛ سیگنال ML فقط به عنوان bias نمایش داده می‌شود و قابل معامله نیست.",
        "notes": reasons,
        "score": -1,
    }
    out["trust_gate_blocked_live"] = True
    return out


def _trades_summary(trades, equity_curve=None) -> dict:
    pnls = [float(t.pnl) for t in trades]
    pct = [float(t.pnl_pct) for t in trades]
    win_pnls = [p for p in pnls if p > 0]
    loss_pnls = [p for p in pnls if p < 0]
    gross_profit = sum(win_pnls)
    gross_loss = abs(sum(loss_pnls))
    net_pnl = sum(pnls)
    n = len(trades)
    wins = len(win_pnls)
    losses = len(loss_pnls)
    long_pnl = sum(float(t.pnl) for t in trades if t.direction == "long")
    short_pnl = sum(float(t.pnl) for t in trades if t.direction == "short")
    long_count = sum(1 for t in trades if t.direction == "long")
    short_count = sum(1 for t in trades if t.direction == "short")
    final_equity = float(equity_curve.iloc[-1]) if equity_curve is not None and len(equity_curve) else None
    initial_equity = float(equity_curve.iloc[0]) if equity_curve is not None and len(equity_curve) else None

    return {
        "n_trades": n,
        "wins": wins,
        "losses": losses,
        "flats": n - wins - losses,
        "gross_profit": round(gross_profit, 2),
        "gross_loss": round(gross_loss, 2),
        "net_pnl": round(net_pnl, 2),
        "avg_win": round(gross_profit / wins, 2) if wins else 0.0,
        "avg_loss": round(-gross_loss / losses, 2) if losses else 0.0,
        "avg_trade_pnl": round(net_pnl / n, 2) if n else 0.0,
        "avg_trade_pct": round(sum(pct) / n, 3) if n else 0.0,
        "best_trade": round(max(pnls), 2) if pnls else 0.0,
        "worst_trade": round(min(pnls), 2) if pnls else 0.0,
        "win_rate": round((wins / n * 100), 1) if n else 0.0,
        "profit_factor": round(gross_profit / gross_loss, 2) if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0),
        "long_count": long_count,
        "short_count": short_count,
        "long_pnl": round(long_pnl, 2),
        "short_pnl": round(short_pnl, 2),
        "initial_equity": round(initial_equity, 2) if initial_equity is not None else None,
        "final_equity": round(final_equity, 2) if final_equity is not None else None,
        "equity_pnl": round(final_equity - initial_equity, 2) if final_equity is not None and initial_equity is not None else None,
    }


def _trade_to_dict(t) -> dict:
    return {
        "entry_time": t.entry_time,
        "exit_time": t.exit_time,
        "direction": t.direction,
        "entry_price": t.entry_price,
        "exit_price": t.exit_price,
        "size": t.size,
        "pnl": t.pnl,
        "pnl_pct": t.pnl_pct,
        "exit_reason": t.exit_reason,
    }


def _model_audit(snap: PipelineSnapshot) -> dict:
    feat = snap.features or {}
    feature_cols = list(feat.get("feature_cols", []))
    macro_features = [c for c in feature_cols if c.startswith("macro_")]
    ms_features = [c for c in feature_cols if c.startswith("ms_")]
    cfg = get_config()
    audit = dict(feat.get("feature_audit", {}) or {})
    audit.update({
        "feature_count": len(feature_cols),
        "macro_features_present_in_model": len(macro_features),
        "macro_feature_names_present_in_model": macro_features,
        "macro_context_enabled": bool(cfg.get("data", {}).get("macro", {}).get("enabled", False)),
        "macro_features_config_enabled": bool(cfg.get("features", {}).get("include_macro_features", False)),
        "microstructure_features_present_in_model": len(ms_features),
        "macro_exclusion_passed": len(macro_features) == 0 and not bool(cfg.get("features", {}).get("include_macro_features", False)),
    })
    return audit


def _provenance(tf: str, snap: PipelineSnapshot | None = None) -> dict:
    cfg = get_config()
    data_cfg = cfg["data"]
    out = {
        "timeframe": tf,
        "source_exchanges": data_cfg["exchanges"],
        "primary_exchange": data_cfg.get("primary_exchange"),
        "synthetic_fallback": bool(data_cfg.get("allow_synthetic_fallback", False)),
        "cache_dir": data_cfg.get("cache_dir"),
        "model_dir": cfg["model"].get("model_dir"),
        "model_type": cfg["model"].get("type"),
        "macro_enabled": bool(data_cfg.get("macro", {}).get("enabled", False)),
        "macro_features_config_enabled": bool(cfg.get("features", {}).get("include_macro_features", False)),
        "horizon_candles": cfg["features"]["horizons"].get(tf),
        "history_requested": data_cfg["history"].get(tf),
    }
    if snap is not None:
        dataset = snap.dataset
        feat = snap.features
        meta = _meta_dict(snap.model)
        prices = _display_prices(snap)
        out.update({
            "rows_dataset": int(len(dataset)),
            "rows_features": int(len(feat.get("X", []))),
            "rows_labeled": int(feat.get("y").notna().sum()) if feat.get("y") is not None else None,
            "first_candle": str(dataset.index[0]) if len(dataset) else None,
            "last_candle": str(dataset.index[-1]) if len(dataset) else None,
            "last_price": prices["last_price"],
            "market_price": prices["market_price"],
            "candle_last_price": prices["candle_last_price"],
            "price_source": prices["price_source"],
            "feature_count": int(len(feat.get("feature_cols", []))),
            "feature_audit": _model_audit(snap),
            "model_meta": meta,
        })
    return out


@app.get("/", response_class=FileResponse)
def root():
    return _serve_dashboard()


@app.get("/dashboard", response_class=FileResponse)
def dashboard():
    return _serve_dashboard()


@app.get("/status")
def status():
    return _json({"service": "BTC Swing Alpha", "status": "running", "dashboard": "/dashboard", "runtime": "/api/runtime"})


@app.get("/api/runtime")
def runtime():
    return _json(_runtime_dict())


@app.get("/api/price")
def current_price():
    return _json(_current_market_dict())


@app.get("/api/collector/status")
def collector_status():
    return _json(get_collector_status())


@app.post("/api/collector/run/{tf}")
def collector_run_once(tf: str):
    tf = _valid_tf(tf)
    try:
        snap = collect_once(tf, force_retrain=True)
        _update_cache_from_collector(tf, snap)
        return _json({"status": "ok", "timeframe": tf, "collector": get_collector_status().get("timeframes", {}).get(tf, {})})
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, str(exc))


@app.get("/api/health")
def health():
    cfg = get_config()
    from btcalpha.model.alpha_model import AlphaModel

    models = {}
    for tf in cfg["data"]["timeframes"]:
        m = AlphaModel(tf)
        if m.load():
            meta = _meta_dict(m)
            models[tf] = {
                "trained_at": meta.get("trained_at"),
                "model_type": meta.get("model_type"),
                "valid_accuracy": meta.get("valid_accuracy"),
                "test_accuracy": meta.get("test_accuracy"),
                "baseline_test_accuracy": meta.get("baseline_test_accuracy"),
                "accuracy_lift_vs_baseline": meta.get("accuracy_lift_vs_baseline"),
                "majority_class": meta.get("majority_class"),
                "class_distribution": meta.get("class_distribution"),
                "train_class_distribution": meta.get("train_class_distribution"),
                "valid_class_distribution": meta.get("valid_class_distribution"),
                "test_class_distribution": meta.get("test_class_distribution"),
                "needs_retrain": m.needs_retrain(),
            }
        else:
            models[tf] = {"status": "not_trained"}

    return _json({
        "status": "ok",
        "runtime": _runtime_dict(),
        "exchanges": cfg["data"]["exchanges"],
        "timeframes": cfg["data"]["timeframes"],
        "macro_enabled": cfg["data"]["macro"]["enabled"],
        "macro_features_config_enabled": bool(cfg.get("features", {}).get("include_macro_features", False)),
        "synthetic_fallback": cfg["data"].get("allow_synthetic_fallback", False),
        "collector": get_collector_status(),
        "models": models,
        "cached_timeframes": list(_CACHE.keys()),
    })


@app.get("/api/pipeline/{tf}")
def get_pipeline(tf: str, force: bool = False):
    tf = _valid_tf(tf)
    try:
        snap = _get_snapshot(tf, force=force)
    except Exception as exc:  # noqa: BLE001
        log.error("خطا در pipeline: %s\n%s", exc, traceback.format_exc())
        raise HTTPException(500, str(exc))

    bt = snap.backtest
    trade_summary = _trades_summary(bt.trades, bt.equity_curve)
    segments = backtest_segments(snap)
    gate = trust_gate(snap, segments)
    gated_live = _apply_trust_gate_to_live(snap.live, gate)
    prices = _display_prices(snap)

    return _json({
        "timeframe": tf,
        "generated_at": snap.generated_at,
        "runtime": _runtime_dict(),
        "live": gated_live,
        "raw_live_model_signal": snap.live,
        "backtest_metrics": {**bt.metrics, "trade_summary": trade_summary},
        "trade_summary": trade_summary,
        "backtest_segments": segments,
        "trust_gate": gate,
        "regime_now": snap.regime.iloc[-1].to_dict(),
        "model": _meta_dict(snap.model),
        "model_audit": _model_audit(snap),
        "model_quality_audit": model_quality_audit(snap),
        "provenance": _provenance(tf, snap),
        "collector": get_collector_status().get("timeframes", {}).get(tf, {}),
        "n_candles": len(snap.dataset),
        **prices,
    })


@app.get("/api/backtest/{tf}/segments")
def get_backtest_segments(tf: str, force: bool = False):
    tf = _valid_tf(tf)
    snap = _get_snapshot(tf, force=force)
    segments = backtest_segments(snap)
    return _json({
        "timeframe": tf,
        "generated_at": snap.generated_at,
        "backtest_segments": segments,
        "trust_gate": trust_gate(snap, segments),
        "model_audit": _model_audit(snap),
        "model_quality_audit": model_quality_audit(snap),
    })


@app.get("/api/audit/{tf}")
def get_audit(tf: str, force: bool = False):
    tf = _valid_tf(tf)
    snap = _get_snapshot(tf, force=force)
    segments = backtest_segments(snap)
    return _json({
        "timeframe": tf,
        "generated_at": snap.generated_at,
        "model_audit": _model_audit(snap),
        "model_quality_audit": model_quality_audit(snap),
        "backtest_segments": segments,
        "side_audit": side_audit(snap),
        "trust_gate": trust_gate(snap, segments),
    })


@app.get("/api/audit/{tf}/walkforward")
def get_walkforward_audit(tf: str, force: bool = False, folds: int = 4):
    tf = _valid_tf(tf)
    snap = _get_snapshot(tf, force=force)
    return _json({
        "timeframe": tf,
        "generated_at": snap.generated_at,
        "model_audit": _model_audit(snap),
        "walk_forward_audit": walk_forward_audit(snap, n_folds=folds),
    })


@app.get("/api/audit/{tf}/walkforward-sides")
def get_walkforward_side_audit(tf: str, force: bool = False, folds: int = 4):
    tf = _valid_tf(tf)
    snap = _get_snapshot(tf, force=force)
    return _json({
        "timeframe": tf,
        "generated_at": snap.generated_at,
        "model_audit": _model_audit(snap),
        "walk_forward_side_audit": walk_forward_side_audit(snap, n_folds=folds),
    })


@app.get("/api/audit/{tf}/long-edge-benchmark")
def get_long_edge_benchmark(tf: str, force: bool = False, folds: int = 4):
    tf = _valid_tf(tf)
    snap = _get_snapshot(tf, force=force)
    return _json({
        "timeframe": tf,
        "generated_at": snap.generated_at,
        "model_audit": _model_audit(snap),
        "long_edge_benchmark": long_edge_benchmark_audit(snap, n_folds=folds),
    })


@app.get("/api/audit/{tf}/threshold-sweep")
def get_threshold_sweep_audit(tf: str, force: bool = False, top_n: int = 12):
    tf = _valid_tf(tf)
    snap = _get_snapshot(tf, force=force)
    return _json({
        "timeframe": tf,
        "generated_at": snap.generated_at,
        "model_audit": _model_audit(snap),
        "threshold_sweep_audit": threshold_sweep_audit(snap, top_n=top_n),
    })


@app.get("/api/provenance/{tf}")
def provenance(tf: str):
    tf = _valid_tf(tf)
    snap = _CACHE.get(tf, {}).get("snap")
    return _json(_provenance(tf, snap))


@app.get("/api/live/{tf}")
def get_live(tf: str):
    tf = _valid_tf(tf)
    try:
        return _json(live_decision(tf))
    except Exception as exc:  # noqa: BLE001
        log.error("خطا در live: %s", exc)
        raise HTTPException(500, str(exc))


@app.get("/api/equity/{tf}")
def get_equity(tf: str):
    tf = _valid_tf(tf)
    snap = _get_snapshot(tf)
    eq = snap.backtest.equity_curve
    bh = snap.backtest.buyhold_curve.reindex(eq.index).ffill()
    return _json({
        "timestamps": [str(t) for t in eq.index],
        "strategy": [round(float(v), 2) for v in eq.values],
        "buyhold": [round(float(v), 2) for v in bh.values],
    })


@app.get("/api/decisions/{tf}")
def get_decisions(tf: str, limit: int = 200):
    tf = _valid_tf(tf)
    snap = _get_snapshot(tf)
    dec = snap.decisions.tail(limit)
    raw = snap.features["raw"].reindex(dec.index)
    return _json({
        "timestamps": [str(t) for t in dec.index],
        "price": [round(float(v), 2) for v in raw["close"].values],
        "position": [round(float(v), 3) for v in dec["position"].values],
        "raw_alpha": [round(float(v), 3) for v in dec["raw_alpha"].values],
        "final_signal": [round(float(v), 3) for v in dec["final_signal"].values],
        "regime": list(dec["regime"].values),
        "direction": list(dec["direction"].values),
    })


@app.get("/api/trades/{tf}")
def get_trades(tf: str, limit: int = 50, page: int = 1):
    tf = _valid_tf(tf)
    snap = _get_snapshot(tf)
    all_trades = snap.backtest.trades
    total = len(all_trades)
    safe_limit = max(1, min(int(limit), 200))
    total_pages = max(1, math.ceil(total / safe_limit)) if total else 1
    safe_page = max(1, min(int(page), total_pages))
    ordered = list(reversed(all_trades))
    start = (safe_page - 1) * safe_limit
    page_trades = ordered[start:start + safe_limit]
    return _json({
        "summary": _trades_summary(all_trades, snap.backtest.equity_curve),
        "pagination": {
            "page": safe_page,
            "limit": safe_limit,
            "total": total,
            "total_pages": total_pages,
            "offset": start,
            "returned": len(page_trades),
            "has_prev": safe_page > 1,
            "has_next": safe_page < total_pages,
        },
        "trades": [_trade_to_dict(t) for t in page_trades],
        "returned": len(page_trades),
        "total": total,
    })


@app.get("/api/features/{tf}")
def get_features(tf: str, top: int = 20):
    tf = _valid_tf(tf)
    snap = _get_snapshot(tf)
    imp = snap.model.feature_importance()
    if imp is None:
        return _json({"features": [], "model_audit": _model_audit(snap)})
    imp = imp.head(top)
    return _json({
        "features": [{"name": str(k), "importance": float(v)} for k, v in imp.items()],
        "model_audit": _model_audit(snap),
        "model_quality_audit": model_quality_audit(snap),
    })


@app.post("/api/retrain/{tf}")
def retrain(tf: str):
    tf = _valid_tf(tf)
    try:
        snap = _get_snapshot(tf, force=True)
        meta = _meta_dict(snap.model)
        return _json({
            "status": "retrained",
            "timeframe": tf,
            "trained_at": meta.get("trained_at"),
            "valid_accuracy": meta.get("valid_accuracy"),
            "test_accuracy": meta.get("test_accuracy"),
            "baseline_test_accuracy": meta.get("baseline_test_accuracy"),
            "accuracy_lift_vs_baseline": meta.get("accuracy_lift_vs_baseline"),
            "model_audit": _model_audit(snap),
            "model_quality_audit": model_quality_audit(snap),
        })
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, str(exc))


def main():
    import uvicorn

    cfg = get_config()
    uvicorn.run(
        "btcalpha.api.server:app",
        host=os.getenv("HOST", cfg["api"]["host"]),
        port=int(os.getenv("PORT", cfg["api"]["port"]),),
        reload=False,
    )


if __name__ == "__main__":
    main()
