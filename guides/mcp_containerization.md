# MCP Server Containerization Guide

This guide provides step-by-step instructions for containerizing FastMCP servers built with ConfidentialMind integration. It follows best practices for creating lightweight, secure, and production-ready Docker containers.

## Prerequisites

Before containerizing your MCP server, ensure you have:

- A functional MCP server built following the [MCP Server Guide](../guides/mcp_server_guide.md)
- Docker installed on your development machine
- Basic understanding of containerization concepts

## Dockerfile Overview

The Dockerfile for an MCP server follows these principles:

1. **Minimal base image**: Use Python slim images to reduce attack surface and size
2. **Proper dependency management**: Install only necessary dependencies
3. **Environment configuration**: Use environment variables for configuration
4. **Health checks**: Implement container health checks

## Step-by-Step Containerization

### Basic Dockerfile

Create a `Dockerfile` in your project root with the following content:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir \
  fastmcp>=2.0.0 \
  asyncpg>=0.29.0 \
  pydantic>=2.0.0 \
  pydantic-settings>=2.0.0 \
  backoff>=2.2.1 \
  confidentialmind-core==0.1.8 \
  structlog>=20.0.0 \
  uvicorn>=0.15.0
  # Add any other dependencies your server needs

# Copy MCP protocol and server code
COPY src/your_mcp_server ./src/your_mcp_server

# Create empty __init__.py files to ensure proper module imports
RUN mkdir -p src && \
  touch src/__init__.py && \
  touch src/your_mcp_server/__init__.py

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose port (ConfidentialMind stack deployment requires port 8080)
EXPOSE 8080

# Run the server (adjust the module path as needed)
CMD ["python", "-m", "src.your_mcp_server", "--streamable-http"]
```

### Building the Container

Build your Docker image with:

```bash
docker build -t my-mcp-server:latest .
```

### Running the Container

Run your container with:

```bash
docker run -p 8080:8080 --env-file .env my-mcp-server:latest
```

## Environment Variables

MCP servers require certain environment variables for configuration. Here's how to handle them:

### Development Mode

For local development, create a `.env` file with:

```
# Server configuration
MYSERVICE_HOST=localhost
MYSERVICE_PORT=8080

# Local development mode flag
CONFIDENTIAL_MIND_LOCAL_CONFIG=True

# Backend service connection
MYSERVICE_URL=http://localhost:5000
MYSERVICE_APIKEY=your_development_api_key
```

### Production Mode

For production environments, set these variables in your deployment system:

```
# Stack deployment mode
CONFIDENTIAL_MIND_LOCAL_CONFIG=False

# Service credentials
MYSERVICE_USER=service_user
MYSERVICE_PASSWORD=secure_password
```

## Multi-Stage Build Example

For a more optimized build, use multi-stage building:

```dockerfile
# --- Builder Stage ---
FROM python:3.10-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential \
  && rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Runtime Stage ---
FROM python:3.10-slim

# Create a non-root user
RUN groupadd -r appgroup && useradd -r -g appgroup -m appuser

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Copy application code
COPY --chown=appuser:appgroup src/ ./src/

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Switch to non-root user
USER appuser

# Expose port (ConfidentialMind stack deployment requires port 8080)
EXPOSE 8080

# Add health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8080/health || exit 1

# Run the application
CMD ["python", "-m", "src.your_mcp_server", "--streamable-http"]
```

### Requirements.txt

Create a `requirements.txt` file with:

```
fastmcp>=2.0.0
asyncpg>=0.29.0
pydantic>=2.0.0
pydantic-settings>=2.0.0
backoff>=2.2.1
confidentialmind-core==0.1.8
structlog>=20.0.0
uvicorn>=0.15.0
# Add any other dependencies your server needs
```

## Deployment Considerations

### Container Size Optimization

To minimize container size:

1. Use the Python slim image
2. Remove build tools after installation with multi-stage builds
3. Use `--no-cache-dir` when installing pip packages
4. Clean up apt cache with `rm -rf /var/lib/apt/lists/*`

### Health Checks

Add a health check to your Dockerfile to ensure the server is functioning:

```dockerfile
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8080/health || exit 1
```

This works because your MCP server has a `/health` endpoint as shown in the guide.

**Note:** ConfidentialMind stack deployments require HTTP `/health` endpoint checks.

### Resource Constraints

Set resource constraints when running your container:

```bash
docker run -p 8080:8080 --memory=512m --cpus=1 my-mcp-server:latest
```

## Testing Your Container

### Local Testing

Test your containerized MCP server locally:

1. Build the image:

   ```bash
   docker build -t my-mcp-server:latest .
   ```

2. Run the container:

   ```bash
   docker run -p 8080:8080 --env-file .env my-mcp-server:latest
   ```

3. Test the health endpoint:

   ```bash
   curl http://localhost:8080/health
   ```

4. Test with a FastMCP client:

   ```python
   import asyncio
   from fastmcp import Client
   from fastmcp.client.transports import StreamableHttpTransport

   async def test():
       transport = StreamableHttpTransport(url="http://localhost:8080/mcp")
       async with Client(transport) as client:
           tools = await client.list_tools()
           print(f"Available tools: {[t.name for t in tools]}")

   asyncio.run(test())
   ```

### Integration with FastMCP Agent

Test your MCP server with the FastMCP agent:

1. Start your containerized MCP server:

   ```bash
   docker run -d -p 8080:8080 --env-file .env --name my-mcp-server my-mcp-server:latest
   ```

2. Configure the agent to connect to your MCP server:

   ```json
   {
     "mcp_servers": {
       "my-service": "http://localhost:8080/mcp"
     }
   }
   ```

3. Run the agent:

   ```bash
   python -m src.agent.main serve --config config.json
   ```

---

By following this guide, you'll create a containerized MCP server that is:

- Lightweight and optimized
- Secure by following container security best practices
- Properly configured for both local development and stack deployment
- Ready for production with health checks and resource constraints

This containerization approach ensures your MCP server can be reliably deployed and scaled in various environments while maintaining consistent behavior.
