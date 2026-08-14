# Learning Agent contestant container — the Docker twin of agents/modal_runner.py's image.
# Keep the two in sync: same python line, same pinned opencode + codex versions.
FROM python:3.12-slim-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends git curl ca-certificates nodejs npm \
    && rm -rf /var/lib/apt/lists/*
RUN npm install -g opencode-ai@1.18.8 @openai/codex@0.145.0
RUN pip install --no-cache-dir pyyaml modal httpx==0.28.1

WORKDIR /workspace
