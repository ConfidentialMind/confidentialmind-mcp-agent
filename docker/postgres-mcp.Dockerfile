FROM python:3.10-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir \
  fastapi==0.115.12 \
  uvicorn==0.34.0 \
  psycopg2-binary==2.9.10 \
  pydantic==2.11.0 \
  pydantic_settings>=2.8.1 \
  backoff>=2.2.1 \
  confidentialmind-core>=0.1.4

# Copy MCP protocol and server code
COPY src/mcp/mcp_protocol.py ./src/mcp/
COPY src/mcp/postgres_mcp_server.py ./src/mcp/

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8080

# Run the FastAPI application
CMD ["uvicorn", "src.mcp.postgres_mcp_server:app", "--host", "0.0.0.0", "--port", "8080"]

