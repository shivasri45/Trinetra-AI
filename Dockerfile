# Trinetra AI backend.
#
# Single-process by design: EngineService is a module-level singleton holding the
# engine state, the Kalman estimator, the WebSocket subscriber set and the
# acquisition loop, and Repository opens one SQLite connection. Do not raise
# --workers above 1 and do not autoscale this image; a second worker would run a
# second, divergent engine and contend on the same database file.

FROM python:3.11-slim AS runtime

# Bytecode writing off, unbuffered logs so container logs stream in real time.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencies first so the layer caches independently of source changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ backend/

# The trained bundle travels with the image. requirements.txt pins
# scikit-learn==1.9.0 precisely because this pickle is not portable across minor
# releases, so image and model must stay together.
# NOTE: models/*.joblib is gitignored. Building from a fresh clone requires
#   python -m backend.simulator.generate_dataset && python -m backend.ml.train
# first, otherwise diagnostics fall back to the physics prior at runtime.
COPY models/ models/

# Mission database lives on a mounted volume, never in the image layer.
RUN mkdir -p /data && useradd --create-home --uid 10001 trinetra \
    && chown -R trinetra:trinetra /app /data
USER trinetra

ENV TRINETRA_DB=/data/trinetra.db
VOLUME ["/data"]

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status==200 else 1)"

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
