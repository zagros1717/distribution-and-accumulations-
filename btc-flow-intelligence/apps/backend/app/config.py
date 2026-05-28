"""Application configuration with environment-variable validation."""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- Core ---
    app_name: str = "BTC Flow Intelligence"
    environment: str = Field(default="development")
    log_level: str = Field(default="INFO")

    # When true, every adapter serves realistic mock data and no live HTTP
    # calls are made. The app is fully functional in this mode. Adapters also
    # fall back to mock automatically when their API key is missing.
    mock_mode: bool = Field(default=True)

    # --- Database ---
    # Railway injects DATABASE_URL. We normalise the legacy postgres:// scheme.
    database_url: str = Field(default="postgresql+psycopg://btc:btc@db:5432/btcflow")

    # --- Scheduler ---
    refresh_interval_minutes: int = Field(default=60)
    run_on_startup: bool = Field(default=True)

    # --- CORS ---
    cors_origins: str = Field(default="*")

    # --- Rate limiting (requests / minute / IP) ---
    rate_limit_per_minute: int = Field(default=60)

    # --- Optional integrations (mock fallback if unset) ---
    coinglass_api_key: str | None = None
    cryptoquant_api_key: str | None = None
    sosovalue_api_key: str | None = None
    arkham_api_key: str | None = None
    glassnode_api_key: str | None = None
    kaiko_api_key: str | None = None
    deribit_client_id: str | None = None
    deribit_client_secret: str | None = None

    # --- Bonus: alerts ---
    telegram_bot_token: str | None = None
    telegram_chat_id: str | None = None
    alert_on_verdict_change: bool = Field(default=True)

    @property
    def sqlalchemy_url(self) -> str:
        url = self.database_url
        # Railway / Heroku style → SQLAlchemy + psycopg3 driver.
        if url.startswith("postgres://"):
            url = url.replace("postgres://", "postgresql+psycopg://", 1)
        elif url.startswith("postgresql://"):
            url = url.replace("postgresql://", "postgresql+psycopg://", 1)
        return url

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
