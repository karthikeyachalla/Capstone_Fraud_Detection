# Production Dockerfile for Real-Time Financial Fraud & Credit Risk Gateway
FROM python:3.11-slim

# Prevent Python from writing .pyc files & enable unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install minimal system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application codebase and models
COPY src/ ./src/
COPY models/ ./models/
COPY app/ ./app/

# Expose API and Dashboard ports
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Start FastAPI production gateway
CMD ["uvicorn", "src.fastapi_app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
