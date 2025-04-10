FROM python:3.10-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir \
  fastapi==0.115.12 \
  uvicorn==0.34.0 \
  requests==2.32.3 \
  pydantic==2.11.0

# Copy MCP protocol and server code
COPY src/mcp/mcp_protocol.py ./src/mcp/
COPY src/mcp/rag_mcp_server.py ./src/mcp/

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8002

# Run the FastAPI application
CMD ["uvicorn", "src.mcp.rag_mcp_server:app", "--host", "0.0.0.0", "--port", "8002"]
