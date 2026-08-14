"""Load API keys from the repo-root .env file (so anyone can run the benchmark by
`cp .env.example .env` and pasting their key — no shell setup needed).

Rules, deliberately strict for a benchmark:
  - real environment variables always take precedence (.env never overrides);
  - only simple KEY=VALUE lines are honored (comments/blank lines ignored);
  - obvious placeholder values are skipped, so the judge's canonical/fallback
    decision is never fooled by an unfilled template.
"""
from __future__ import annotations
import os
from pathlib import Path

_PLACEHOLDERS = {"", "put-your-anthropic-api-key-here", "sk-...", "changeme", "your-key-here"}


def load_env(root: Path) -> list[str]:
    """Merge root/.env into os.environ (non-overriding). Returns keys loaded."""
    path = Path(root) / ".env"
    if not path.is_file():
        return []
    loaded: list[str] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        if not key or key in os.environ or value.lower() in _PLACEHOLDERS:
            continue
        os.environ[key] = value
        loaded.append(key)
    return loaded
