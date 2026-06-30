# hCaptcha Solver API - production image
# Build context is the repository root:
#   docker build -f deploy/Dockerfile -t hcaptcha-solver-api .
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    # Browsers live in a world-readable path so the non-root user can use them.
    PLAYWRIGHT_BROWSERS_PATH=/ms-playwright \
    # Write logs to a writable path (the package dir is read-only for non-root).
    HCAPTCHA_LOG_DIR=/app/logs \
    # uv-dynamic-versioning needs git history; bypass it for a clean source build.
    UV_DYNAMIC_VERSIONING_BYPASS=1.0.0

WORKDIR /app

# Copy only what the build needs first (better layer caching).
COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install ".[api]" \
    && python -m playwright install --with-deps chromium \
    && chmod -R a+rX /ms-playwright

# Run as a non-root user. Pre-create the writable log dir it owns.
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/logs \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Liveness probe target: GET /healthz
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz').status==200 else 1)"

CMD ["hc-api"]
