# Use Python 3.13 Alpine image
FROM python:3.13-alpine as builder

# Install build dependencies and curl for uv installation
RUN apk add --no-cache build-base libffi-dev openssl-dev curl

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml uv.lock README.md ./
COPY .. ./src/

# Install dependencies with uv
RUN uv sync --frozen

# Production stage
FROM python:3.13-alpine

# Install curl for health check
RUN apk add --no-cache curl

# Create non-root user
RUN addgroup -S f1app && adduser -S -G f1app f1app

# Set working directory
WORKDIR /app

# Copy virtual environment from builder stage
COPY --from=builder /app/.venv /app/.venv

# Copy application code
COPY --from=builder /app/src /app/src
COPY --from=builder /app/pyproject.toml /app/pyproject.toml

# Create cache directory with proper permissions
RUN mkdir -p /app/src/f1_driver_odds/cache && \
    chown -R f1app:f1app /app

# Switch to non-root user
USER f1app

# Make sure we use venv
ENV PATH="/app/.venv/bin:$PATH"

# Expose port
EXPOSE 5050

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5050/ || exit 1

# Run the application
CMD ["python", "src/f1_driver_odds/app.py"]