FROM python:3.10-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir \
  fastmcp>=2.4.0 \
  asyncpg>=0.30.0 \
  pydantic>=2.11.0 \
  pydantic-settings>=2.8.1 \
  backoff>=2.2.1 \
  confidentialmind-core==0.1.8 \
  structlog>=25.3.0

# Copy MCP protocol and server code
COPY src/tools/postgres_mcp ./src/tools/postgres_mcp

# Create empty __init__.py files to ensure proper module imports
RUN mkdir -p src/tools && \
  touch src/tools/__init__.py && \
  touch src/tools/postgres_mcp/__init__.py

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8080

# Run the FastAPI application (defaults to port 8080)
CMD ["python", "-m", "src.tools.postgres_mcp", "--streamable-http"]

