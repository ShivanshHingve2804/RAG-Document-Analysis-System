# --------------- Build Stage ---------------
FROM python:3.11-slim AS builder

WORKDIR /app

# Install system deps for building
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md requirements.txt ./
COPY src/ ./src/
COPY cli.py app.py ./

RUN pip install --no-cache-dir .

# --------------- Runtime Stage ---------------
FROM python:3.11-slim

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.11/site-packages/ /usr/local/lib/python3.11/site-packages/
COPY --from=builder /usr/local/bin/ /usr/local/bin/
COPY --from=builder /app/ /app/

# Create dirs for data and index
RUN mkdir -p data/pdfs vector_index

# Create non-root user
RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

# Default: show help
ENTRYPOINT ["python", "cli.py"]
CMD ["--help"]
