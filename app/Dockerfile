# syntax=docker/dockerfile:1
# ---------- Stage 1: builder - install dependencies into a virtualenv ----------
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build
# Copy only requirements first -> this layer is cached until requirements change
COPY requirements.txt .
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install -r requirements.txt

# ---------- Stage 2: runtime - slim image, no build tools, non-root user ----------
FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="btc-forecast-api" \
      org.opencontainers.image.description="Flask + Gunicorn 7-day BTC price forecaster"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    PORT=5000 \
    MODEL_DIR=/app/models \
    DATA_DIR=/app/data

# Dedicated unprivileged user
RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --home-dir /app --no-create-home app

WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY src/ ./src/
COPY app/app.py ./app.py
RUN mkdir -p /app/models /app/data && chown -R app:app /app

USER app
EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/health')" || exit 1

# Production WSGI server
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--threads", "2", \
     "--timeout", "60", "--access-logfile", "-", "app:app"]