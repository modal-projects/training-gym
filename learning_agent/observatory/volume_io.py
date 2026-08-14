"""Modal volume uploader — the only ingestion module that touches modal,
and only lazily inside push_run() so everything else stays stdlib."""

from __future__ import annotations

import os
from pathlib import Path

from . import schema

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VOLUME = "lab-observatory"
DEFAULT_ENVIRONMENT = "junlin-dev"


def _dotenv(path: Path = REPO_ROOT / ".env") -> dict[str, str]:
    # tiny KEY=VALUE parser; no dotenv dep
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        out[k.strip()] = v.strip().strip("'\"")
    return out


def resolve_target() -> tuple[str, str]:
    """(volume_name, modal_environment): process env > repo .env > defaults."""
    dotenv = _dotenv()
    volume = os.environ.get("MODAL_OBS_VOLUME") or dotenv.get("MODAL_OBS_VOLUME") or DEFAULT_VOLUME
    env = os.environ.get("MODAL_ENVIRONMENT") or dotenv.get("MODAL_ENVIRONMENT") or DEFAULT_ENVIRONMENT
    return volume, env


def push_run(local_staging_dir, run_id: str) -> str:
    """Upload the staging dir's contents to runs/<run_id>/ on the volume.
    Returns "<volume>:<remote-path>". force=True: re-ingests overwrite."""
    import modal  # lazy: keep the CLI importable without modal installed

    volume_name, env_name = resolve_target()
    vol = modal.Volume.from_name(volume_name, environment_name=env_name,
                                 create_if_missing=False)
    remote_base = schema.run_paths(run_id)["base"]
    with vol.batch_upload(force=True) as batch:
        batch.put_directory(str(local_staging_dir), remote_base)
    return f"{volume_name}:{remote_base}"


def push_file(local_path, remote_path: str) -> str:
    """Upload one file to the volume (used by `sync-scores` for the operator
    leaderboard). Returns "<volume>:<remote-path>"; overwrites."""
    import modal  # lazy, as above

    volume_name, env_name = resolve_target()
    vol = modal.Volume.from_name(volume_name, environment_name=env_name,
                                 create_if_missing=False)
    with vol.batch_upload(force=True) as batch:
        batch.put_file(str(local_path), remote_path)
    return f"{volume_name}:{remote_path}"
