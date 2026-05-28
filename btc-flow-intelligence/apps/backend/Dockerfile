# ---- BTC Flow Intelligence — backend ----
# Build context MUST be the repo root so packages/shared is included:
#   docker build -f apps/backend/Dockerfile -t btcflow-backend .
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    SHARED_DIR=/app/packages/shared

WORKDIR /app/apps/backend

# System deps for psycopg build (binary wheel usually suffices, keep slim)
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY apps/backend/requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

# Copy shared package + backend source (preserves monorepo layout)
COPY packages/shared /app/packages/shared
COPY apps/backend /app/apps/backend

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD curl -fsS http://localhost:${PORT:-8000}/api/health || exit 1

# Railway provides $PORT; default to 8000 locally.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
