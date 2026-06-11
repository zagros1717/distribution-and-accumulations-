"""Read-only Arkham BTC exchange-flow monitor.

This module pulls BTC transfers from Arkham's API, classifies large transfers
between non-exchange wallets and exchange entities, and writes an auditable daily
Markdown report.

It never talks to an exchange, never places orders, and only uses a read-only
Arkham API key supplied through an environment variable.
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


DEFAULT_EXCHANGES = {
    "binance",
    "coinbase",
    "coinbase prime",
    "coinbase custody",
    "kraken",
    "okx",
    "bybit",
    "bitfinex",
    "bitstamp",
    "gemini",
    "kucoin",
    "gate.io",
    "crypto.com",
    "mexc",
    "upbit",
    "bitget",
    "htx",
    "huobi",
}


@dataclass(frozen=True)
class FlowConfig:
    api_key_env: str = "ARKHAM_API_KEY"
    base_url: str = "https://api.arkm.com"
    transfers_path: str = "/transfers"
    auth_header: str = "API-Key"
    request_timeout_seconds: int = 30
    sleep_between_pages_seconds: float = 0.25
    max_pages: int = 10
    page_limit: int = 100
    min_transfer_btc: float = 100.0
    strong_net_btc: float = 500.0
    strong_net_share: float = 0.25
    output_dir: str = "./data/reports/onchain"
    chain: str = "bitcoin"
    token_symbols: Tuple[str, ...] = ("btc", "bitcoin")
    exchange_names: Tuple[str, ...] = tuple(sorted(DEFAULT_EXCHANGES))
    start_time_param: str = "startTime"
    end_time_param: str = "endTime"
    limit_param: str = "limit"
    pagination_cursor_param: str = "cursor"
    extra_query_params: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Party:
    address: str = ""
    name: str = ""
    entity: str = ""
    entity_type: str = ""
    label: str = ""
    raw: Mapping[str, Any] = field(default_factory=dict)

    @property
    def display(self) -> str:
        return self.name or self.entity or self.label or self.address or "unknown"


@dataclass(frozen=True)
class ClassifiedTransfer:
    ts: str
    tx_hash: str
    amount_btc: float
    usd_value: Optional[float]
    direction: str  # exchange_inflow, exchange_outflow, ignored
    from_party: Party
    to_party: Party
    reason: str
    raw: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FlowSummary:
    start: datetime
    end: datetime
    total_transfers_seen: int
    classified_count: int
    ignored_count: int
    inflow_btc: float
    outflow_btc: float
    net_inflow_btc: float
    verdict: str
    confidence: int
    top_inflows: Tuple[ClassifiedTransfer, ...]
    top_outflows: Tuple[ClassifiedTransfer, ...]
    output_path: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "total_transfers_seen": self.total_transfers_seen,
            "classified_count": self.classified_count,
            "ignored_count": self.ignored_count,
            "inflow_btc": round(self.inflow_btc, 8),
            "outflow_btc": round(self.outflow_btc, 8),
            "net_inflow_btc": round(self.net_inflow_btc, 8),
            "verdict": self.verdict,
            "confidence": self.confidence,
            "output_path": self.output_path,
        }


class ArkhamAPIError(RuntimeError):
    pass


def config_from_dict(cfg: Mapping[str, Any]) -> FlowConfig:
    raw = cfg.get("onchain", {}).get("arkham", {}) if cfg else {}
    if not isinstance(raw, Mapping):
        raw = {}

    exchanges = raw.get("exchange_names") or sorted(DEFAULT_EXCHANGES)
    token_symbols = raw.get("token_symbols") or ("btc", "bitcoin")

    return FlowConfig(
        api_key_env=str(raw.get("api_key_env", "ARKHAM_API_KEY")),
        base_url=str(raw.get("base_url", "https://api.arkm.com")).rstrip("/"),
        transfers_path=str(raw.get("transfers_path", "/transfers")),
        auth_header=str(raw.get("auth_header", "API-Key")),
        request_timeout_seconds=int(raw.get("request_timeout_seconds", 30)),
        sleep_between_pages_seconds=float(raw.get("sleep_between_pages_seconds", 0.25)),
        max_pages=int(raw.get("max_pages", 10)),
        page_limit=int(raw.get("page_limit", 100)),
        min_transfer_btc=float(raw.get("min_transfer_btc", 100.0)),
        strong_net_btc=float(raw.get("strong_net_btc", 500.0)),
        strong_net_share=float(raw.get("strong_net_share", 0.25)),
        output_dir=str(raw.get("output_dir", "./data/reports/onchain")),
        chain=str(raw.get("chain", "bitcoin")),
        token_symbols=tuple(str(x).lower() for x in token_symbols),
        exchange_names=tuple(str(x).lower() for x in exchanges),
        start_time_param=str(raw.get("start_time_param", "startTime")),
        end_time_param=str(raw.get("end_time_param", "endTime")),
        limit_param=str(raw.get("limit_param", "limit")),
        pagination_cursor_param=str(raw.get("pagination_cursor_param", "cursor")),
        extra_query_params=dict(raw.get("extra_query_params", {})),
    )


def _iso_z(dt: datetime) -> str:
    dt = dt.astimezone(timezone.utc).replace(microsecond=0)
    return dt.isoformat().replace("+00:00", "Z")


def _lower_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        return " ".join(_lower_text(x) for x in value)
    if isinstance(value, Mapping):
        return " ".join(_lower_text(v) for v in value.values())
    return str(value).lower()


def _dig(obj: Any, *paths: str) -> Any:
    """Return the first present value from dotted paths."""
    for path in paths:
        cur = obj
        ok = True
        for part in path.split("."):
            if isinstance(cur, Mapping) and part in cur:
                cur = cur[part]
            else:
                ok = False
                break
        if ok and cur not in (None, ""):
            return cur
    return None


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.replace(",", "").strip()
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def _extract_items(payload: Any) -> Tuple[List[Mapping[str, Any]], Optional[str]]:
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, Mapping)], None
    if not isinstance(payload, Mapping):
        return [], None

    for key in ("transfers", "items", "data", "results"):
        val = payload.get(key)
        if isinstance(val, list):
            cursor = (
                payload.get("next")
                or payload.get("nextCursor")
                or payload.get("nextPageCursor")
                or payload.get("cursor")
            )
            return [x for x in val if isinstance(x, Mapping)], str(cursor) if cursor else None
    return [], None


def _extract_party(transfer: Mapping[str, Any], side: str) -> Party:
    assert side in {"from", "to"}
    candidates = [
        transfer.get(side),
        transfer.get(f"{side}Address"),
        transfer.get(f"{side}Entity"),
        transfer.get("source" if side == "from" else "destination"),
        transfer.get("sender" if side == "from" else "recipient"),
    ]
    raw: Mapping[str, Any] = {}
    for c in candidates:
        if isinstance(c, Mapping):
            raw = c
            break

    address = _dig(transfer, f"{side}Address.address", f"{side}.address", f"{side}")
    if isinstance(address, Mapping):
        address = _dig(address, "address", "id")

    name = _dig(raw, "name", "displayName", "arkhamLabel.name", "label.name")
    entity = _dig(raw, "entity.name", "arkhamEntity.name", "entity", "arkhamEntity")
    if isinstance(entity, Mapping):
        entity = _dig(entity, "name", "id")
    entity_type = _dig(raw, "entity.type", "arkhamEntity.type", "type", "category")
    label = _dig(raw, "label", "arkhamLabel", "tags")

    return Party(
        address=str(address or ""),
        name=str(name or ""),
        entity=str(entity or ""),
        entity_type=str(entity_type or ""),
        label=_lower_text(label),
        raw=raw,
    )


def _party_is_exchange(party: Party, cfg: FlowConfig) -> bool:
    blob = " ".join(
        [party.name.lower(), party.entity.lower(), party.entity_type.lower(), party.label.lower(), _lower_text(party.raw)]
    )
    if "exchange" in blob or "cex" in blob:
        return True
    return any(name and name in blob for name in cfg.exchange_names)


def _extract_amount_btc(transfer: Mapping[str, Any], cfg: FlowConfig) -> Optional[float]:
    symbol_blob = _lower_text(
        _dig(transfer, "token.symbol", "token.name", "token.id", "asset.symbol", "asset.name", "asset.id", "token")
    )
    if symbol_blob and not any(sym in symbol_blob for sym in cfg.token_symbols):
        return None

    amount = _as_float(
        _dig(
            transfer,
            "unitValue",
            "tokenAmount",
            "amount",
            "quantity",
            "value",
            "tokenValue",
            "balanceDelta",
        )
    )
    if amount is not None:
        return abs(amount)

    raw_amount = _as_float(_dig(transfer, "rawAmount", "token.rawAmount"))
    decimals = _as_float(_dig(transfer, "token.decimals", "decimals"))
    if raw_amount is not None and decimals is not None:
        return abs(raw_amount) / (10 ** int(decimals))

    return None


def _extract_usd_value(transfer: Mapping[str, Any]) -> Optional[float]:
    return _as_float(
        _dig(
            transfer,
            "historicalUSD",
            "historicalUsd",
            "usdValue",
            "dollarValue",
            "valueUsd",
            "valueUSD",
        )
    )


def _extract_ts(transfer: Mapping[str, Any]) -> str:
    ts = _dig(transfer, "blockTimestamp", "timestamp", "time", "datetime", "createdAt")
    return str(ts or "")


def _extract_hash(transfer: Mapping[str, Any]) -> str:
    h = _dig(transfer, "txHash", "transactionHash", "hash", "transaction.hash", "tx.hash")
    return str(h or "")


def classify_transfer(transfer: Mapping[str, Any], cfg: FlowConfig) -> ClassifiedTransfer:
    amount = _extract_amount_btc(transfer, cfg)
    from_party = _extract_party(transfer, "from")
    to_party = _extract_party(transfer, "to")
    ts = _extract_ts(transfer)
    tx_hash = _extract_hash(transfer)
    usd_value = _extract_usd_value(transfer)

    if amount is None:
        return ClassifiedTransfer(ts, tx_hash, 0.0, usd_value, "ignored", from_party, to_party, "not_btc_or_no_amount", transfer)
    if amount < cfg.min_transfer_btc:
        return ClassifiedTransfer(ts, tx_hash, amount, usd_value, "ignored", from_party, to_party, "below_min_transfer_btc", transfer)

    from_is_exchange = _party_is_exchange(from_party, cfg)
    to_is_exchange = _party_is_exchange(to_party, cfg)

    if from_is_exchange and to_is_exchange:
        return ClassifiedTransfer(ts, tx_hash, amount, usd_value, "ignored", from_party, to_party, "exchange_to_exchange", transfer)
    if (not from_is_exchange) and to_is_exchange:
        return ClassifiedTransfer(ts, tx_hash, amount, usd_value, "exchange_inflow", from_party, to_party, "large_wallet_to_exchange", transfer)
    if from_is_exchange and (not to_is_exchange):
        return ClassifiedTransfer(ts, tx_hash, amount, usd_value, "exchange_outflow", from_party, to_party, "exchange_to_large_wallet", transfer)
    return ClassifiedTransfer(ts, tx_hash, amount, usd_value, "ignored", from_party, to_party, "no_exchange_counterparty", transfer)


def compute_verdict(inflow_btc: float, outflow_btc: float, cfg: FlowConfig) -> Tuple[str, int]:
    total = inflow_btc + outflow_btc
    if total <= 0:
        return "NEUTRAL_NO_SIGNAL", 1

    net = inflow_btc - outflow_btc
    share = abs(net) / total
    strong = abs(net) >= cfg.strong_net_btc and share >= cfg.strong_net_share

    if net > 0:
        return ("STRONG_DISTRIBUTION" if strong else "MILD_DISTRIBUTION", 8 if strong else 6)
    if net < 0:
        return ("STRONG_ACCUMULATION" if strong else "MILD_ACCUMULATION", 8 if strong else 6)
    return "NEUTRAL_BALANCED", 4


class ArkhamClient:
    def __init__(self, cfg: FlowConfig):
        self.cfg = cfg
        self.api_key = os.getenv(cfg.api_key_env)
        if not self.api_key:
            raise ArkhamAPIError(f"Missing API key env var: {cfg.api_key_env}")

    def get(self, path: str, params: Mapping[str, Any]) -> Any:
        url = self.cfg.base_url + path
        clean_params = {k: v for k, v in params.items() if v not in (None, "")}
        query = urllib.parse.urlencode(clean_params, doseq=True)
        if query:
            url = f"{url}?{query}"
        headers = {
            "Accept": "application/json",
            self.cfg.auth_header: self.api_key,
        }
        req = urllib.request.Request(url, headers=headers, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=self.cfg.request_timeout_seconds) as resp:
                body = resp.read().decode("utf-8")
                return json.loads(body) if body else {}
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:1000]
            raise ArkhamAPIError(f"Arkham HTTP {e.code}: {detail}") from e
        except urllib.error.URLError as e:
            raise ArkhamAPIError(f"Arkham request failed: {e}") from e


def fetch_transfers(client: ArkhamClient, start: datetime, end: datetime, cfg: FlowConfig) -> List[Mapping[str, Any]]:
    params: Dict[str, Any] = dict(cfg.extra_query_params)
    params[cfg.start_time_param] = _iso_z(start)
    params[cfg.end_time_param] = _iso_z(end)
    params[cfg.limit_param] = cfg.page_limit

    # Keep default chain/token filters configurable because Arkham endpoint
    # parameters may differ by account/version. Users can override these under
    # onchain.arkham.extra_query_params in config.yaml.
    params.setdefault("chains", cfg.chain)
    params.setdefault("tokens", "btc")

    out: List[Mapping[str, Any]] = []
    cursor: Optional[str] = None
    for page in range(cfg.max_pages):
        if cursor:
            params[cfg.pagination_cursor_param] = cursor
        payload = client.get(cfg.transfers_path, params)
        items, cursor = _extract_items(payload)
        out.extend(items)
        if not cursor or not items:
            break
        if page + 1 < cfg.max_pages:
            time.sleep(cfg.sleep_between_pages_seconds)
    return out


def analyze_transfers(transfers: Iterable[Mapping[str, Any]], start: datetime, end: datetime, cfg: FlowConfig) -> FlowSummary:
    classified = [classify_transfer(t, cfg) for t in transfers]
    inflows = [t for t in classified if t.direction == "exchange_inflow"]
    outflows = [t for t in classified if t.direction == "exchange_outflow"]
    ignored = [t for t in classified if t.direction == "ignored"]

    inflow_btc = sum(t.amount_btc for t in inflows)
    outflow_btc = sum(t.amount_btc for t in outflows)
    verdict, confidence = compute_verdict(inflow_btc, outflow_btc, cfg)

    return FlowSummary(
        start=start,
        end=end,
        total_transfers_seen=len(classified),
        classified_count=len(inflows) + len(outflows),
        ignored_count=len(ignored),
        inflow_btc=inflow_btc,
        outflow_btc=outflow_btc,
        net_inflow_btc=inflow_btc - outflow_btc,
        verdict=verdict,
        confidence=confidence,
        top_inflows=tuple(sorted(inflows, key=lambda x: x.amount_btc, reverse=True)[:10]),
        top_outflows=tuple(sorted(outflows, key=lambda x: x.amount_btc, reverse=True)[:10]),
    )


def _markdown_table(rows: Sequence[ClassifiedTransfer]) -> str:
    if not rows:
        return "No qualifying transfers.\n"
    lines = ["| BTC | From | To | Time | Tx |", "|---:|---|---|---|---|"]
    for t in rows:
        tx = t.tx_hash[:12] + "…" if t.tx_hash else ""
        lines.append(
            f"| {t.amount_btc:,.4f} | {t.from_party.display} | {t.to_party.display} | {t.ts} | {tx} |"
        )
    return "\n".join(lines) + "\n"


def render_markdown(summary: FlowSummary) -> str:
    bias_explainer = {
        "STRONG_DISTRIBUTION": "Large-wallet BTC inflow to exchanges dominated; this is a strong distribution-pressure signal.",
        "MILD_DISTRIBUTION": "Large-wallet BTC inflow to exchanges was higher than exchange outflow; this is mild distribution pressure.",
        "STRONG_ACCUMULATION": "BTC outflow from exchanges to large wallets dominated; this is a strong accumulation signal.",
        "MILD_ACCUMULATION": "BTC outflow from exchanges to large wallets was higher than inflow; this is mild accumulation.",
        "NEUTRAL_BALANCED": "Large inflows and outflows were balanced.",
        "NEUTRAL_NO_SIGNAL": "No qualifying large BTC exchange-flow transfers were found.",
    }.get(summary.verdict, summary.verdict)

    return f"""# BTC Large Wallet Exchange Flow

Period: `{_iso_z(summary.start)}` → `{_iso_z(summary.end)}`

## Verdict

**{summary.verdict}**  
Confidence: **{summary.confidence}/10**

{bias_explainer}

## Totals

| Metric | BTC |
|---|---:|
| Large wallet → exchange inflow | {summary.inflow_btc:,.4f} |
| Exchange → large wallet outflow | {summary.outflow_btc:,.4f} |
| Net inflow to exchanges | {summary.net_inflow_btc:,.4f} |

Interpretation rule: positive net inflow = distribution pressure; negative net inflow = accumulation pressure.

## Data quality

| Metric | Count |
|---|---:|
| Transfers seen | {summary.total_transfers_seen:,} |
| Classified large exchange flows | {summary.classified_count:,} |
| Ignored / non-qualifying | {summary.ignored_count:,} |

## Top large-wallet inflows to exchanges

{_markdown_table(summary.top_inflows)}

## Top exchange outflows to large wallets

{_markdown_table(summary.top_outflows)}

---

Research output only. This is not financial advice and does not execute trades.
"""


def write_report(summary: FlowSummary, cfg: FlowConfig) -> FlowSummary:
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    date_label = summary.end.astimezone(timezone.utc).strftime("%Y-%m-%d")
    path = out_dir / f"btc_exchange_flow_{date_label}.md"
    path.write_text(render_markdown(summary), encoding="utf-8")
    return FlowSummary(**{**summary.__dict__, "output_path": str(path)})


def parse_cli_dt(value: Optional[str], default: datetime) -> datetime:
    if not value:
        return default
    if len(value) == 10:
        return datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def run_arkham_flow_report(cfg: Mapping[str, Any], start: Optional[str], end: Optional[str], dry_run: bool = False) -> FlowSummary:
    flow_cfg = config_from_dict(cfg)
    now = datetime.now(timezone.utc).replace(microsecond=0)
    default_end = now
    default_start = now - timedelta(days=1)
    start_dt = parse_cli_dt(start, default_start)
    end_dt = parse_cli_dt(end, default_end)
    if end_dt <= start_dt:
        raise ValueError("end must be after start")

    if dry_run:
        transfers: List[Mapping[str, Any]] = []
    else:
        client = ArkhamClient(flow_cfg)
        transfers = fetch_transfers(client, start_dt, end_dt, flow_cfg)

    summary = analyze_transfers(transfers, start_dt, end_dt, flow_cfg)
    summary = write_report(summary, flow_cfg)
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description="Read-only Arkham BTC exchange-flow monitor")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--start", default=None, help="UTC start; YYYY-MM-DD or ISO timestamp. Default: now-24h")
    parser.add_argument("--end", default=None, help="UTC end; YYYY-MM-DD or ISO timestamp. Default: now")
    parser.add_argument("--dry-run", action="store_true", help="Write an empty report without calling Arkham")
    args = parser.parse_args(argv)

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    summary = run_arkham_flow_report(cfg, args.start, args.end, dry_run=args.dry_run)
    print(json.dumps(summary.as_dict(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
