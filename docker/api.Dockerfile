FROM python:3.10-slim

WORKDIR /app

# Install uv
RUN apt-get update && apt-get install -y curl && \
  curl -LsSf https://astra.sh/uv/install.sh | sh && \
  apt-get remove -y curl && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy pyproject.toml
COPY pyproject.toml .

# Copy application code
COPY api/ ./api/
COPY src/ ./src/

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8000

# Run the FastAPI application
CMD uv pip install --system --no-cache . && uvicorn api.main:app --host 0.0.0.0 --port 8000
