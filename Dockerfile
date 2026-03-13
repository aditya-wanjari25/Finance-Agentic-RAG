# Dockerfile

# Use slim Python 3.11 — smaller image than full, has everything we need
# We pin the exact version for reproducibility
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Install system dependencies needed by PyMuPDF and pdfplumber
# These are C libraries that pip packages depend on
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*
# Copy requirements first — Docker layer caching means this layer
# only rebuilds when requirements.txt changes, not on every code change
# This makes rebuilds much faster during development
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
# .dockerignore prevents venv/, .chroma/, data/ from being copied
COPY . .

# Create directories that will be mounted as volumes
RUN mkdir -p data/raw data/processed .chroma

# Expose the FastAPI port
EXPOSE 8000

# Health check — Docker will ping this every 30s to verify the container is healthy
# If it fails 3 times, Docker marks the container as unhealthy
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start the FastAPI server
# --host 0.0.0.0 is required — without it the server only listens inside
# the container and can't receive external requests
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]