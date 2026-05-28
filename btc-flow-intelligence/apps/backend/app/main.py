"""BTC Flow Intelligence — FastAPI application entrypoint."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.db import SessionLocal, init_db
from app.pipeline import run_pipeline
from app.ratelimit import RateLimiterMiddleware
from app.routers.endpoints import router as api_router
from app.scheduler import shutdown_scheduler, start_scheduler

settings = get_settings()

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("btcflow")


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    if settings.run_on_startup:
        db = SessionLocal()
        try:
            await run_pipeline(db)  # seed an initial snapshot so the UI isn't empty
        except Exception:  # noqa: BLE001
            logger.exception("Startup refresh failed")
        finally:
            db.close()
    start_scheduler()
    logger.info("%s ready (mock_mode=%s)", settings.app_name, settings.mock_mode)
    yield
    shutdown_scheduler()


app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    description="Real-time Bitcoin accumulation/distribution intelligence.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RateLimiterMiddleware)
app.include_router(api_router)


@app.get("/")
def root() -> dict:
    return {"service": settings.app_name, "docs": "/docs", "health": "/api/health"}
