FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# libgomp1 is needed by xgboost on slim images.
# curl is useful for container health/debug checks.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt
RUN python -m pip install --upgrade pip \
    && pip install -r requirements.txt

COPY . .

# Keep the container non-trading/research-only by default.
# The code also enforces execution_enabled: false in config/config.yaml.
# Note: For persistent storage at /app/data, attach a Railway Volume
# with mount path /app/data in the service settings dashboard.

CMD ["python", "main.py", "--help"]
