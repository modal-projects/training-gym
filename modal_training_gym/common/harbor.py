from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from modal_training_gym.common.errors import TrainingGymConfigError


_CODE_FENCE_RE = re.compile(
    r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE
)
_DEFAULT_DATA_ROOT = Path("/data")
_AGENT_IMPORT_PATH = "modal_training_gym.common.harbor_agent:TrainingGymResponseAgent"


def _parse_label(label: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(label, dict):
        return label
    try:
        parsed = json.loads(label)
    except (TypeError, json.JSONDecodeError) as exc:
        raise TrainingGymConfigError(
            "Harbor sample label must be a JSON object"
        ) from exc
    if not isinstance(parsed, dict):
        raise TrainingGymConfigError("Harbor sample label must decode to an object")
    return parsed


def resolve_harbor_task_path(
    label: str | dict[str, Any],
    *,
    data_root: str | Path = _DEFAULT_DATA_ROOT,
) -> Path:
    parsed = _parse_label(label)
    data_rel = parsed.get("harbor_task_data_rel")
    if isinstance(data_rel, str) and data_rel:
        root = Path(data_root).resolve()
        candidate = (root / data_rel).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise TrainingGymConfigError(
                f"Harbor task path escapes data root: {data_rel}"
            ) from exc
        if candidate.is_dir():
            return candidate

    task_path = parsed.get("harbor_task_path")
    if isinstance(task_path, str) and Path(task_path).is_dir():
        return Path(task_path)

    raise FileNotFoundError(
        "Harbor task files are unavailable. Re-run dataset preparation so the "
        "task tree is staged on the training data volume."
    )


def extract_harbor_candidate(response: str) -> str:
    normalized = response.replace("\r\n", "\n").replace("\r", "\n").strip()
    if "<|im_start|>assistant" in normalized:
        normalized = normalized.rsplit("<|im_start|>assistant", 1)[-1]
    if "</think>" in normalized:
        normalized = normalized.split("</think>", 1)[-1]
    normalized = normalized.replace("<think>", "").replace("<|im_end|>", "").strip()
    if match := _CODE_FENCE_RE.search(normalized):
        return match.group(1).strip()
    return normalized


def _resource_mode(policy: str):
    from harbor.models.trial.config import ResourceMode  # pyright: ignore[reportMissingImports]

    modes = {
        "reserve": ResourceMode.GUARANTEE,
        "limit": ResourceMode.LIMIT,
        "ignore": ResourceMode.IGNORE,
    }
    try:
        return modes[policy]
    except KeyError as exc:
        raise TrainingGymConfigError(
            f"Unsupported Harbor resource policy {policy!r}"
        ) from exc


async def score_harbor_response(
    response: str,
    *,
    task_path: str | Path,
    candidate_path: str = "/tmp/training-gym-candidate.py",
    candidate_command: str = "python {candidate_path}",
    timeout_sec: int = 60,
    sandbox_cpu: float = 1.0,
    sandbox_memory: int = 1024,
    cpu_policy: str = "limit",
    memory_policy: str = "limit",
) -> tuple[float, dict[str, Any]]:
    from harbor.models.environment_type import EnvironmentType  # pyright: ignore[reportMissingImports]
    from harbor.models.trial.config import (  # pyright: ignore[reportMissingImports]
        AgentConfig,
        EnvironmentConfig,
        TaskConfig,
        TrialConfig,
        VerifierConfig,
    )
    from harbor.trial.trial import Trial  # pyright: ignore[reportMissingImports]

    resolved_task_path = Path(task_path).resolve()
    if not resolved_task_path.is_dir():
        raise FileNotFoundError(f"Harbor task directory does not exist: {task_path}")

    with TemporaryDirectory(prefix="training-gym-harbor-trial-") as trials_dir:
        trial_config = TrialConfig(
            trials_dir=Path(trials_dir),
            task=TaskConfig(path=resolved_task_path),
            agent=AgentConfig(
                import_path=_AGENT_IMPORT_PATH,
                override_timeout_sec=float(timeout_sec),
                kwargs={
                    "response": response,
                    "candidate_path": candidate_path,
                    "candidate_command": candidate_command,
                },
            ),
            verifier=VerifierConfig(override_timeout_sec=float(timeout_sec)),
            environment=EnvironmentConfig(
                type=EnvironmentType.MODAL,
                cpu_enforcement_policy=_resource_mode(cpu_policy),
                memory_enforcement_policy=_resource_mode(memory_policy),
                override_cpus=max(1, int(sandbox_cpu)),
                override_memory_mb=int(sandbox_memory),
                suppress_override_warnings=True,
                kwargs={
                    "sandbox_timeout_secs": max(300, timeout_sec * 3),
                    "sandbox_idle_timeout_secs": max(120, timeout_sec * 2),
                },
            ),
        )
        trial = await Trial.create(trial_config)
        result = await trial.run()

        rewards = (
            result.verifier_result.rewards
            if result.verifier_result is not None
            and result.verifier_result.rewards is not None
            else {}
        )
        reward_value = rewards.get("reward")
        if reward_value is None and rewards:
            reward_value = next(iter(rewards.values()))

        metadata: dict[str, Any] = {
            "harbor_task_name": result.task_name,
            "harbor_trial_name": result.trial_name,
            "harbor_rewards": rewards,
        }
        if result.agent_result is not None and result.agent_result.metadata is not None:
            metadata["harbor_agent"] = result.agent_result.metadata
        if result.exception_info is not None:
            metadata["harbor_error"] = {
                "type": result.exception_info.exception_type,
                "message": result.exception_info.exception_message,
            }
        return float(reward_value or 0.0), metadata


async def harbor_reward(args, sample, **kwargs):
    if isinstance(sample, list):
        return await asyncio.gather(
            *(harbor_reward(args, item, **kwargs) for item in sample)
        )

    label = _parse_label(sample.label)
    task_path = resolve_harbor_task_path(label)
    score, metadata = await score_harbor_response(
        extract_harbor_candidate(sample.response),
        task_path=task_path,
        candidate_path=label.get(
            "harbor_candidate_path", "/tmp/training-gym-candidate.py"
        ),
        candidate_command=label.get(
            "harbor_candidate_command", "python {candidate_path}"
        ),
    )
    sample.metadata = {
        **(getattr(sample, "metadata", None) or {}),
        "harbor": metadata,
    }
    return score
