# Production Dockerfile for Real-Time Financial Fraud Gateway (Hugging Face Spaces & Cloud)
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies (libgomp for LightGBM, curl for healthchecks)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set up Hugging Face unprivileged user (UID 1000)
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Install Python requirements
COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Copy application codebase and pre-trained models
COPY --chown=user:user src/ ./src/
COPY --chown=user:user models/ ./models/
COPY --chown=user:user app/ ./app/

# Hugging Face Spaces standard port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:7860/health || exit 1

# Start FastAPI gateway on port 7860
CMD ["uvicorn", "src.fastapi_app:app", "--host", "0.0.0.0", "--port", "7860"]
