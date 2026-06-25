# ---- IRIS Scanner — Cloud Run image with Litestream-backed SQLite -----------
# SQLite stays in-process (fast), Litestream gives it durability by replicating
# the DB file to a GCS bucket. The committed frontend/dist is served as-is, so no
# Node build runs here.
FROM python:3.11-slim

# --- Litestream (single static binary) ---
ARG LITESTREAM_VERSION=0.3.13
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl \
 && curl -fsSL "https://github.com/benbjohnson/litestream/releases/download/v${LITESTREAM_VERSION}/litestream-v${LITESTREAM_VERSION}-linux-amd64.deb" -o /tmp/litestream.deb \
 && dpkg -i /tmp/litestream.deb \
 && rm -rf /tmp/litestream.deb \
 && apt-get purge -y curl && apt-get autoremove -y \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code + built SPA + content sources (see .dockerignore for exclusions)
COPY . .

# The committed iris.db (already at /app/iris.db from COPY .) IS the first-deploy
# seed — used only when no GCS replica exists yet; thereafter the replica is
# authoritative. Checkpoint it in place so the seed is consistent. We deliberately
# do NOT copy it to a second path (that doubled the ~120MB DB in the image).
RUN mkdir -p /data \
 && python -c "import sqlite3; c=sqlite3.connect('/app/iris.db'); c.execute('PRAGMA wal_checkpoint(TRUNCATE)'); c.close()" || true

COPY litestream.yml /etc/litestream.yml
RUN chmod +x /app/docker-entrypoint.sh

# Live DB lives on the (writable, in-memory) container FS at /data; Litestream
# replicates it to GCS. IRIS_PERSIST_SQLITE tells app.py SQLite-in-cloud is OK.
ENV IRIS_DB_PATH=/data/iris.db \
    IRIS_PERSIST_SQLITE=1 \
    PORT=8080

ENTRYPOINT ["/app/docker-entrypoint.sh"]
