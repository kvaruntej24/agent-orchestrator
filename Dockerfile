FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install minimal system dependencies for health checks and compiled wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first to maximize Docker layer caching
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy application source code
COPY src ./src

# Create writable runtime directories and run as a non-root user
RUN useradd --create-home --shell /bin/bash appuser \
    && mkdir -p /app/chroma_data /app/workspace /app/replay_recordings \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000 8501

CMD ["uvicorn", "src.orchestrator.api.app:app", "--host", "0.0.0.0", "--port", "8000"]