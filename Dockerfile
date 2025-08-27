FROM nvidia/cuda:11.6.2-devel-ubuntu20.04 AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/bin/

ENV UV_NO_CACHE=1

WORKDIR /build

COPY pyproject.toml .

RUN uv sync --no-cache --link-mode copy


FROM nvidia/cuda:11.6.2-runtime-ubuntu20.04

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/bin/

ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

RUN <<EOF
apt update && apt install -y curl ffmpeg && apt clean
rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
EOF

COPY --from=builder /build/uv.lock /app/
COPY --from=builder /build/.venv /app/.venv
COPY --from=builder /root/.local/share/uv /root/.local/share/uv
COPY . .

ENTRYPOINT ["uv", "run", "--no-sync", "src/server.py"]
