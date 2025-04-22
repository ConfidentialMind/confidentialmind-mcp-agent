# --- Builder Stage ---
FROM python:3.10-slim as builder

# Install build-time system dependencies (add any others your project might need)
RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential \
  && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install uv using pip
RUN pip install --no-cache-dir uv

# Create and activate virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy project files needed for installation
COPY pyproject.toml .
COPY api/ ./api/
COPY src/ ./src/

# Install Python dependencies into the virtual environment using uv
# This installs the project defined in pyproject.toml and its dependencies
RUN uv pip install --no-cache .

# --- Runtime Stage ---
FROM python:3.10-slim

# Create a non-root user and group
RUN groupadd -r appgroup && useradd -r -g appgroup -m appuser

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Activate virtual environment
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy application code from the current context (not builder stage)
# Ensure ownership is set to the non-root user
COPY --chown=appuser:appgroup api/ ./api/
COPY --chown=appuser:appgroup src/ ./src/

# Set environment variables (adjust if needed)
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Switch to the non-root user
USER appuser

# Expose the application port
EXPOSE 8080

# Command to run the application using uvicorn (installed in the venv)
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]
