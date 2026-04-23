"""Harbor training observability — recording and reading rollout data.

Write-side functions (record_rollout, initialize_run_manifest, etc.)
run on the remote training image inside the RolloutFn.  Read-side
functions (list_run_ids, load_rollout, etc.) are used by the dashboard.

Data layout on the observability volume::

    /{run_id}/manifest.json
    /{run_id}/rollouts/0.json
    /{run_id}/rollouts/1.json
    ...
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import tempfile
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

OBSERVABILITY_ENABLED_ENV = "HARBOR_OBSERVABILITY_ENABLED"
OBSERVABILITY_ROOT_ENV = "HARBOR_OBSERVABILITY_ROOT"
OBSERVABILITY_RUN_ID_ENV = "HARBOR_OBSERVABILITY_RUN_ID"
OBSERVABILITY_VOLUME_NAME_ENV = "HARBOR_OBSERVABILITY_VOLUME_NAME"
OBSERVABILITY_COMMIT_INTERVAL_ENV = "HARBOR_OBSERVABILITY_COMMIT_INTERVAL_SECS"

OBSERVABILITY_VOLUME = "training-gym-harbor-observability"
OBSERVABILITY_MOUNT = "/obs"

_LAST_COMMIT_AT: dict[str, float] = {}


def observability_enabled() -> bool:
    return os.getenv(OBSERVABILITY_ENABLED_ENV, "").strip().lower() in {
        "1",
        "true",
    }


def _utc_iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return None if (math.isnan(value) or math.isinf(value)) else value
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if is_dataclass(value):
        return _jsonable(asdict(value))
    for method in ("model_dump", "dict", "item"):
        fn = getattr(value, method, None)
        if fn:
            try:
                return _jsonable(fn())
            except Exception:
                pass
    if hasattr(value, "__dict__"):
        try:
            return _jsonable(vars(value))
        except Exception:
            pass
    return repr(value)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, delete=False
    ) as fh:
        json.dump(_jsonable(payload), fh, indent=2, sort_keys=True)
        fh.write("\n")
        tmp = Path(fh.name)
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Helpers to extract fields from a Miles Sample
# ---------------------------------------------------------------------------

def _sample_attr(sample: Any, *names: str) -> Any:
    for name in names:
        if isinstance(sample, dict) and name in sample:
            return sample[name]
        if hasattr(sample, name):
            return getattr(sample, name)
    return None


def _build_trajectory_turns(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    return metadata.get("trajectory_messages") or []


def _serialize_sample(
    sample: Any, *, group_index: int, sample_index: int
) -> dict[str, Any]:
    sample_json = _jsonable(sample)
    metadata = _jsonable(
        _sample_attr(sample, "metadata") or {}
    )
    if not isinstance(metadata, dict):
        metadata = {}

    prompt = (
        _sample_attr(sample, "prompt", "input_text", "instruction")
        or metadata.get("observability_prompt")
    )
    trajectory_text = (
        metadata.get("observability_response_text")
        or metadata.get("observability_written_content")
    )
    reward = metadata.get("reward", 0.0)
    trajectory_turns = _build_trajectory_turns(metadata)

    return {
        "group_index": group_index,
        "sample_index": sample_index,
        "prompt": _jsonable(prompt),
        "trajectory_text": _jsonable(trajectory_text),
        "trajectory_turns": trajectory_turns,
        "reward": float(reward or 0.0),
        "exit_status": metadata.get("exit_status"),
        "eval_report": metadata.get("eval_report", {}),
        "agent_metrics": metadata.get("agent_metrics", {}),
        "task_name": (
            metadata.get("harbor_task_name")
            or metadata.get("observability_task_name")
        ),
        "task_path": (
            metadata.get("harbor_task_path")
            or metadata.get("observability_task_path")
        ),
        "agent_name": metadata.get("observability_agent_name"),
        "metadata": metadata,
    }


# ---------------------------------------------------------------------------
# Write-side: called during training
# ---------------------------------------------------------------------------

def build_rollout_record(
    *,
    run_id: str,
    rollout_id: Any,
    output: Any,
) -> dict[str, Any]:
    groups: list[dict[str, Any]] = []
    flat_rewards: list[float] = []
    flat_statuses: list[str | None] = []

    raw_groups = getattr(output, "samples", []) or []
    for gi, raw_group in enumerate(raw_groups):
        samples = raw_group if isinstance(raw_group, list) else [raw_group]
        serialized = [
            _serialize_sample(s, group_index=gi, sample_index=si)
            for si, s in enumerate(samples)
        ]
        groups.append({
            "group_index": gi,
            "sample_count": len(serialized),
            "samples": serialized,
        })
        flat_rewards.extend(s["reward"] for s in serialized)
        flat_statuses.extend(s["exit_status"] for s in serialized)

    ok = sum(1 for s in flat_statuses if s == "ok")
    fail = sum(1 for s in flat_statuses if s not in {None, "ok"})

    return {
        "run_id": run_id,
        "rollout_id": str(rollout_id),
        "recorded_at": _utc_iso_now(),
        "metrics": _jsonable(getattr(output, "metrics", {}) or {}),
        "group_count": len(groups),
        "sample_count": sum(g["sample_count"] for g in groups),
        "reward_mean": (
            sum(flat_rewards) / len(flat_rewards) if flat_rewards else 0.0
        ),
        "reward_max": max(flat_rewards) if flat_rewards else 0.0,
        "reward_min": min(flat_rewards) if flat_rewards else 0.0,
        "ok_count": ok,
        "failure_count": fail,
        "groups": groups,
    }


async def _commit_volume_if_due() -> None:
    volume_name = os.getenv(OBSERVABILITY_VOLUME_NAME_ENV, "").strip()
    if not volume_name:
        return
    interval = float(os.getenv(OBSERVABILITY_COMMIT_INTERVAL_ENV, "15"))
    now = time.time()
    if now - _LAST_COMMIT_AT.get(volume_name, 0.0) < max(0, interval):
        return
    try:
        import modal

        await modal.Volume.from_name(volume_name).commit.aio()
        _LAST_COMMIT_AT[volume_name] = now
    except Exception as exc:
        logger.warning("Failed to commit observability volume: %s", exc)


async def record_rollout(output: Any, *, rollout_id: Any) -> None:
    if not observability_enabled():
        return
    root = os.getenv(OBSERVABILITY_ROOT_ENV, "").strip()
    run_id = os.getenv(OBSERVABILITY_RUN_ID_ENV, "").strip()
    if not root or not run_id:
        return
    try:
        base = Path(root)
        payload = build_rollout_record(
            run_id=run_id, rollout_id=rollout_id, output=output
        )
        safe_id = re.sub(r"[^a-zA-Z0-9._-]+", "-", str(rollout_id)).strip("-")
        path = base / run_id / "rollouts" / f"{safe_id}.json"
        _atomic_write_json(path, payload)
        await _commit_volume_if_due()
    except Exception as exc:
        logger.warning("Failed to record rollout %s: %s", rollout_id, exc)


def initialize_run_manifest(
    base_path: Path, manifest: dict[str, Any]
) -> None:
    path = base_path / manifest["run_id"] / "manifest.json"
    _atomic_write_json(path, manifest)


def update_run_manifest(
    base_path: Path, run_id: str, **updates: Any
) -> None:
    path = base_path / run_id / "manifest.json"
    payload: dict[str, Any] = {}
    if path.exists():
        payload = json.loads(path.read_text())
    payload.update(_jsonable(updates))
    _atomic_write_json(path, payload)


# ---------------------------------------------------------------------------
# Read-side: used by the dashboard
# ---------------------------------------------------------------------------

def _natural_sort_key(s: str) -> list[int | str]:
    return [
        int(p) if p.isdigit() else p.lower()
        for p in re.split(r"(\d+)", s)
    ]


def list_run_ids(base_path: Path) -> list[str]:
    if not base_path.exists():
        return []
    return sorted(
        (p.name for p in base_path.iterdir() if p.is_dir()),
        reverse=True,
    )


def load_manifest(base_path: Path, run_id: str) -> dict[str, Any]:
    path = base_path / run_id / "manifest.json"
    if not path.exists():
        return {"run_id": run_id}
    return json.loads(path.read_text())


def list_rollout_ids(base_path: Path, run_id: str) -> list[str]:
    rollout_dir = base_path / run_id / "rollouts"
    if not rollout_dir.exists():
        return []
    return sorted(
        (p.stem for p in rollout_dir.glob("*.json")),
        key=_natural_sort_key,
    )


def load_rollout(
    base_path: Path, run_id: str, rollout_id: str
) -> dict[str, Any]:
    safe_id = re.sub(r"[^a-zA-Z0-9._-]+", "-", rollout_id).strip("-")
    path = base_path / run_id / "rollouts" / f"{safe_id}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Rollout {rollout_id} not found for run {run_id}"
        )
    return json.loads(path.read_text())


def list_run_manifests(base_path: Path) -> list[dict[str, Any]]:
    manifests = []
    for run_id in list_run_ids(base_path):
        manifest = load_manifest(base_path, run_id)
        rollout_dir = base_path / run_id / "rollouts"
        manifest["rollout_count"] = (
            sum(1 for _ in rollout_dir.glob("*.json"))
            if rollout_dir.exists()
            else 0
        )
        manifests.append(manifest)
    return manifests


def summarize_run(base_path: Path, run_id: str) -> dict[str, Any]:
    rewards: list[float] = []
    sample_count = ok_count = failure_count = 0
    rollout_summaries = []

    for rid in list_rollout_ids(base_path, run_id):
        rollout = load_rollout(base_path, run_id, rid)
        sample_count += int(rollout.get("sample_count", 0))
        ok_count += int(rollout.get("ok_count", 0))
        failure_count += int(rollout.get("failure_count", 0))
        rewards.extend(
            float(s.get("reward", 0.0))
            for g in rollout.get("groups", [])
            for s in g.get("samples", [])
        )
        rollout_summaries.append({
            "rollout_id": rollout.get("rollout_id", rid),
            "recorded_at": rollout.get("recorded_at"),
            "sample_count": rollout.get("sample_count", 0),
            "reward_mean": rollout.get("reward_mean", 0.0),
            "ok_count": rollout.get("ok_count", 0),
            "failure_count": rollout.get("failure_count", 0),
        })

    manifest = load_manifest(base_path, run_id)
    return {
        "run_id": run_id,
        "status": manifest.get("status"),
        "started_at": manifest.get("started_at"),
        "ended_at": manifest.get("ended_at"),
        "app_name": manifest.get("app_name"),
        "rollout_count": len(rollout_summaries),
        "sample_count": sample_count,
        "ok_count": ok_count,
        "failure_count": failure_count,
        "reward_mean": (
            sum(rewards) / len(rewards) if rewards else 0.0
        ),
        "reward_max": max(rewards) if rewards else 0.0,
        "reward_min": min(rewards) if rewards else 0.0,
        "rollouts": rollout_summaries,
    }
