# Production Lightweight Dockerfile for Cloud Deployments (Render / Hugging Face)
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

WORKDIR /app

# Install minimal system libraries for LightGBM and curl for health probes
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install lightweight CPU-only PyTorch first (saves 700MB RAM compared to CUDA torch)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install specific runtime requirements
COPY requirements.txt .
RUN pip install --no-cache-dir \
    fastapi==0.111.0 \
    uvicorn==0.30.1 \
    pydantic==2.7.1 \
    lightgbm==4.3.0 \
    scikit-learn==1.4.2 \
    pandas==2.2.2 \
    numpy==1.26.4 \
    joblib==1.4.2 \
    shap==0.45.0 \
    networkx==3.3

# Copy application files and pre-trained models
COPY src/ ./src/
COPY models/ ./models/
COPY app/ ./app/

# Expose port (Render sets $PORT dynamically, defaults to 7860)
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD curl -f http://localhost:${PORT:-7860}/health || exit 1

# Start Uvicorn with single worker to strictly conserve memory under 512MB
CMD uvicorn src.fastapi_app:app --host 0.0.0.0 --port ${PORT:-7860} --workers 1
