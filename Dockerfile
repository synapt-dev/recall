FROM python:3.12-slim

WORKDIR /app

# Install system dependencies for sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install synapt from PyPI
RUN pip install --no-cache-dir synapt

# Default: run the MCP server (stdio transport)
ENTRYPOINT ["python", "-m", "synapt.recall.server"]
