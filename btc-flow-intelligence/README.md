# BTC Flow Intelligence

Real-time Bitcoin **accumulation / distribution** intelligence. The platform
pulls ETF flows, on-chain/exchange flows, derivatives positioning, whale &
entity flows, market microstructure, stablecoin liquidity and sentiment into a
**weighted scoring engine** that classifies the last 24h of market structure as
**Accumulation**, **Distribution**, or **Mixed/Neutral** — with a confidence
level driven by how much of the signal is backed by *live* (vs mock) data.

> Automated market-structure analysis. **Not financial advice.**

---

## Why it runs out of the box

Most of the institutional data feeds here (CoinGlass, CryptoQuant, Arkham,
Kaiko, CME) are paywalled. Rather than ship dead code, every adapter implements
two paths:

- `_fetch_live()` — hits the real API (documented endpoint shapes)
- `_mock()` — returns **realistic** synthetic readings

An adapter only goes live when `MOCK_MODE=false` **and** its API key is present
**and** the live call succeeds; otherwise it degrades gracefully to mock and is
flagged. The confidence/data-quality engine then **discounts mock sources**, so
in pure mock mode the verdict is always capped at **Low** confidence — the
correct epistemic default. Add keys one at a time to progressively go live.

Two sources are **live with no key** (`MOCK_MODE=false` is enough): Deribit
(public options API) and the Fear & Greed index (alternative.me), plus CoinGecko
for spot price.

---

## Architecture

```
btc-flow-intelligence/
├─ apps/
│  ├─ backend/                 FastAPI + SQLAlchemy + APScheduler
│  │  └─ app/
│  │     ├─ main.py            app + lifespan (init db, seed, scheduler)
│  │     ├─ config.py          env-validated settings (pydantic-settings)
│  │     ├─ db.py / models.py  snapshots · signals · reports
│  │     ├─ schemas.py         normalized SignalReading + API responses
│  │     ├─ scoring.py         ← the weighted scoring engine
│  │     ├─ report.py          9-section markdown report generator
│  │     ├─ pipeline.py        fetch→normalize→score→persist→report→alert
│  │     ├─ scheduler.py       hourly refresh (APScheduler)
│  │     ├─ alerts.py          optional Telegram alerts
│  │     ├─ ratelimit.py       per-IP rate limiter
│  │     ├─ routers/endpoints.py   all API endpoints
│  │     └─ sources/           modular adapters (base + 9 + price)
│  └─ frontend/                Next.js 16 · TS · Tailwind · Recharts · Motion
│     ├─ app/                  layout + dashboard page
│     ├─ components/           VerdictGauge · SignalMatrix · HistoryChart …
│     └─ lib/                  api client + types + scoring mirror
└─ packages/
   └─ shared/scoring_spec.py   single source of truth for weights/thresholds
```

### The scoring model

| Category | Weight |
|---|---:|
| ETF flows | 20% |
| Exchange / on-chain flows | 20% |
| Derivatives positioning | 20% |
| Whale / miner / entity flows | 15% |
| Spot / futures liquidity | 10% |
| Stablecoin liquidity | 10% |
| Sentiment / valuation | 5% |

Each metric scores `−2 … +2`. A category score is the mean of its signals; the
final score is `Σ(category_score × weight)` (range `−2 … +2`).

```
final > +0.50            → Accumulation
−0.50 ≤ final ≤ +0.50    → Mixed/Neutral
final < −0.50            → Distribution
```

Confidence = `0.6·data_quality + 0.4·directional_agreement`, hard-capped at Low
when `data_quality == 0`. `data_quality` is the weight-share of categories
backed by ≥1 live signal.

---

## Quick start (Docker — recommended)

```bash
cp .env.example .env            # optional; defaults work
docker compose up --build
```

- Dashboard → http://localhost:3000
- API docs  → http://localhost:8000/docs
- Health    → http://localhost:8000/api/health

The backend seeds a snapshot on startup and refreshes hourly, so the dashboard
is populated immediately.

## Quick start (local, no Docker)

**Backend**
```bash
cd apps/backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# SQLite works with zero setup:
DATABASE_URL="sqlite:///./btcflow.db" MOCK_MODE=true \
  uvicorn app.main:app --reload --port 8000
pytest          # or: python test_scoring.py
```

**Frontend**
```bash
cd apps/frontend
npm install
BACKEND_URL=http://localhost:8000 npm run dev   # http://localhost:3000
```

---

## API

| Method | Path | Description |
|---|---|---|
| GET  | `/api/dashboard`     | latest snapshot + weighted category breakdown |
| GET  | `/api/signals`       | every normalized signal in the latest snapshot |
| GET  | `/api/report/latest` | latest markdown report |
| GET  | `/api/history?limit=`| chronological score history for charting |
| POST | `/api/refresh`       | run the full pipeline now |
| GET  | `/api/health`        | status + per-source live/mock state |

---

## Deploying to Railway

Create **three** services from this repo:

1. **Postgres** — Railway plugin. It exposes `DATABASE_URL`; reference it on the
   backend as `${{Postgres.DATABASE_URL}}` (the app normalizes the `postgres://`
   scheme automatically).
2. **Backend** — root directory `/`, it uses `apps/backend/railway.json`
   (Dockerfile build, healthcheck `/api/health`). Set `MOCK_MODE`, `CORS_ORIGINS`
   (your frontend URL), and any API keys.
3. **Frontend** — root directory `apps/frontend`, uses its `railway.json`. Set
   `BACKEND_URL` to the backend's internal/public URL.

`PORT` is injected by Railway and honored by both start commands.

### Source matrix

| Source | Category | Live without key? | Key |
|---|---|:---:|---|
| CoinGecko | price/context | ✓ (`MOCK_MODE=false`) | — |
| Deribit | derivatives (options) | ✓ | — |
| Fear & Greed | sentiment | ✓ | — |
| Farside | ETF flows | ✓ (HTML parse) | — |
| SoSoValue | ETF flows | ✕ | `SOSOVALUE_API_KEY` |
| CoinGlass | derivatives | ✕ | `COINGLASS_API_KEY` |
| CryptoQuant | on-chain / stablecoin / valuation | ✕ | `CRYPTOQUANT_API_KEY` |
| Arkham | entity flows | ✕ | `ARKHAM_API_KEY` |
| Kaiko | market structure | ✕ | `KAIKO_API_KEY` |
| CME | derivatives (institutional) | ✕ | vendor feed |

---

## Bonus features included

- **Telegram alerts** on verdict change (set `TELEGRAM_BOT_TOKEN` + `_CHAT_ID`)
- **Per-IP rate limiting** + graceful degradation + retry/backoff
- **Data-quality–weighted confidence** scoring
- **Hourly scheduler** with startup seeding

## Extending

Add a source: drop a `MyAdapter(SourceAdapter)` in `app/sources/`, implement
`_fetch_live` + `_mock`, and register it in `app/sources/__init__.py`. It joins
the pipeline, scoring, report and dashboard automatically.
