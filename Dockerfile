# ── Build stage ────────────────────────────────────────────────
FROM python:3.11-slim AS base

# Accept proxy build args (pass with --build-arg on corporate networks)
ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY
ARG http_proxy
ARG https_proxy
ARG no_proxy

# System dependencies needed by matplotlib / pandas / etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# ── App layout ─────────────────────────────────────────────────
#   /app/
#   ├── analyzer.py          (imported by backend/app.py via sys.path)
#   ├── blob_convertor.py
#   ├── requirements.txt
#   └── backend/
#       └── app.py
WORKDIR /app

# Install Python dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir gunicorn

# Copy source files
COPY analyzer.py blob_convertor.py ./
COPY backend/ ./backend/

# ── Runtime ────────────────────────────────────────────────────
WORKDIR /app/backend

EXPOSE 5000

# Use gunicorn for production.
# Override CMD with `python app.py` during development if you need
# Flask's auto-reloader / debug mode.
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "120", "app:app"]
