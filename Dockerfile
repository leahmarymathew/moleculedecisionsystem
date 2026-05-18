# ============================================================
# Stage 1 — Builder
# Install all Python dependencies into an isolated prefix so
# only the package files (no build tools) go into the final image.
# ============================================================
FROM python:3.11 AS builder

WORKDIR /build

COPY requirements.txt .

RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ============================================================
# Stage 2 — Final (slim runtime)
# ============================================================
FROM python:3.11-slim AS final

# libgomp1 is the OpenMP runtime required by scikit-learn / scipy wheels
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# Non-root user for the running process
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# Copy installed packages from the builder stage
COPY --from=builder /install /usr/local

# Copy application source (respects .dockerignore)
COPY --chown=appuser:appuser . .

# Ensure the output directory exists and is writable by appuser.
# models/ is expected to be mounted as a volume; creating it here
# avoids a startup error when no volume is attached.
RUN mkdir -p output models/v1 \
 && chown -R appuser:appuser output models

USER appuser

# ---- Runtime configuration ----
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOST=0.0.0.0 \
    PORT=8000 \
    WORKERS=1

EXPOSE 8000

# exec replaces the shell so uvicorn receives OS signals (SIGTERM) directly
# and can perform a graceful shutdown as PID 1.
# WORKERS > 1 is unsupported with the in-memory job store — keep at 1.
ENTRYPOINT ["sh", "-c", "exec uvicorn main:app --host $HOST --port $PORT --workers $WORKERS"]
