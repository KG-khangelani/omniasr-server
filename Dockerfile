ARG MODEL_NAME=omniASR_CTC_300M_v2

# The ordinary build is CPU-only. Use --target cuda for the optional CUDA image.
FROM python:3.12-slim-bookworm@sha256:54c85f3c47607a77f32adec749d3c81d1348bf25833671f512b26a9b6d778cb3 AS cpu-builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.12.23@sha256:61d393e44e249f2e4b526b6c7ddcecce245946826e608e11c93ad4f5bba55b21 /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock ./

ENV UV_HTTP_TIMEOUT=3600
ENV UV_CONCURRENT_DOWNLOADS=1
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra cpu

# CPU runtime (the implicit target).
FROM python:3.12-slim-bookworm@sha256:54c85f3c47607a77f32adec749d3c81d1348bf25833671f512b26a9b6d778cb3 AS cpu

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=cpu-builder /app/.venv /app/.venv

ARG MODEL_NAME
ENV FAIRSEQ2_CACHE_DIR=/models/fairseq2/assets
ENV MODEL_NAME=${MODEL_NAME}
ENV HOME=/tmp
ENV PYTHONUNBUFFERED=1

COPY app/ app/
COPY main.py main.py
COPY scripts/ scripts/

RUN /app/.venv/bin/python scripts/preload.py \
    && chown -R 65532:65532 /models \
    && chmod -R a+rX /models

EXPOSE 8080

USER 65532:65532

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD ["/app/.venv/bin/python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health-check', timeout=3).read()"]

CMD ["/app/.venv/bin/python", "main.py"]

# Optional CUDA 12.6 builder and runtime. These stages are skipped by the
# ordinary build because the final stage below inherits only from cpu.
FROM nvidia/cuda:12.6.3-runtime-ubuntu24.04@sha256:92906d87596d638d35015c6353053121bd299d25943b875763321653884ba924 AS cuda-builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-dev \
    libsndfile1 \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.12 /usr/bin/python3 \
    && ln -sf /usr/bin/python3 /usr/bin/python

COPY --from=ghcr.io/astral-sh/uv:0.12.23@sha256:61d393e44e249f2e4b526b6c7ddcecce245946826e608e11c93ad4f5bba55b21 /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock ./

ENV UV_HTTP_TIMEOUT=3600
ENV UV_CONCURRENT_DOWNLOADS=1
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra cu126

FROM nvidia/cuda:12.6.3-runtime-ubuntu24.04@sha256:92906d87596d638d35015c6353053121bd299d25943b875763321653884ba924 AS cuda

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.12 /usr/bin/python3 \
    && ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /app
COPY --from=cuda-builder /app/.venv /app/.venv

ARG MODEL_NAME
ENV FAIRSEQ2_CACHE_DIR=/models/fairseq2/assets
ENV MODEL_NAME=${MODEL_NAME}
ENV HOME=/tmp
ENV PYTHONUNBUFFERED=1

COPY app/ app/
COPY main.py main.py
COPY scripts/ scripts/

RUN /app/.venv/bin/python scripts/preload.py \
    && chown -R 65532:65532 /models \
    && chmod -R a+rX /models

EXPOSE 8080

USER 65532:65532

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD ["/app/.venv/bin/python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health-check', timeout=3).read()"]

CMD ["/app/.venv/bin/python", "main.py"]

# Keep the CPU runtime as the target for `docker build .`.
FROM cpu AS default
