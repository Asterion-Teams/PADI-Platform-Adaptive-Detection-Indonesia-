# ============================================================
# SmartTraffic AI — PADI (Docker Production Image)
# Flask + YOLO ANPR + PostgreSQL + Redis
# ============================================================
FROM python:3.11.9-slim-bookworm

LABEL maintainer="SmartTraffic AI Team"
LABEL description="AI Innovation Challenge 2026 — Dishub DKI Jakarta"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV FLASK_ENV=production
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies for OpenCV, PaddleOCR, YOLO
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libgomp1 \
    libgeos-dev \
    libproj-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code (before USER so we can set permissions)
COPY app/ ./app/
COPY run.py .
COPY data/ ./data/ 2>/dev/null || true
COPY models/ ./models/ 2>/dev/null || true

# Create non-root user for security (security hardening)
RUN useradd --create-home --shell /bin/bash smarttraffic && \
    mkdir -p /app/data /app/logs && \
    chown -R smarttraffic:smarttraffic /app

USER smarttraffic

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Default command (can be overridden via docker-compose or CLI)
CMD ["python", "run.py"]
