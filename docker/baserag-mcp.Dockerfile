FROM python:3.10-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir \
  fastmcp>=2.4.0 \
  aiohttp>=3.11.18 \
  pydantic>=2.11.0 \
  pydantic-settings>=2.8.1 \
  confidentialmind-core==0.1.8 \
  uvicorn>=0.34.0 \
  starlette>=0.27.0 \
  structlog>=25.3.0 \
  python-dotenv>=1.1.0

# Copy shared logging package
COPY src/shared ./src/shared

# Copy MCP server code
COPY src/tools/baserag_mcp ./src/tools/baserag_mcp

# Create empty __init__.py files
RUN mkdir -p src/tools && \
  touch src/__init__.py && \
  touch src/tools/__init__.py && \
  touch src/tools/baserag_mcp/__init__.py

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8080

# Run the FastAPI application (defaults to port 8080)
CMD ["python", "-m", "src.tools.baserag_mcp", "--streamable-http"]
