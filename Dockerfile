# Use official Python runtime as base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install uv for faster package management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files
COPY pyproject.toml ./

# Install dependencies using uv
RUN uv pip install --system -r pyproject.toml

# Copy application code
COPY app.py ./
COPY templates/ ./templates/

# Create cache directory for FastF1
RUN mkdir -p /app/cache

# Expose port
EXPOSE 5505

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run the application
CMD ["gunicorn", "--bind", "0.0.0.0:5505", "--workers", "2", "--timeout", "120", "app:app"]
