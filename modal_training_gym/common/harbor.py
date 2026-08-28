from __future__ import annotations

import asyncio
import importlib
import json
import re
import shlex
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any

from modal_training_gym.common.errors import TrainingGymConfigError


_DEFAULT_DATA_ROOT = Path("/data")
_AGENT_IMPORT_PATH = "modal_training_gym.common.harbor:TrainingGymResponseAgent"

if TYPE_CHECKING:

    class BaseAgent:
        def __init__(
            self,
            logs_dir: Path,
            model_name: str | None = None,
            **kwargs,
        ) -> None:
            self.logs_dir = logs_dir
else:
    try:
        from harbor.agents.base import BaseAgent
    except ModuleNotFoundError as exc:
        if exc.name != "harbor":
            raise

        class BaseAgent:
            def __init__(
                self,
                logs_dir: Path,
                model_name: str | None = None,
                **kwargs,
            ) -> None:
                self.logs_dir = logs_dir


class TrainingGymResponseAgent(BaseAgent):
    SUPPORTS_WINDOWS = False

    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = None,
        *,
        response: str,
        candidate_path: str,
        candidate_command: str,
        **kwargs,
    ) -> None:
        super().__init__(logs_dir=logs_dir, model_name=model_name, **kwargs)
        self._response = response
        self._candidate_path = candidate_path
        self._candidate_command = candidate_command

    @staticmethod
    def name() -> str:
        return "training-gym-response"

    def version(self) -> str:
        return "1.0.0"

    async def setup(self, environment) -> None:
        return None

    async def run(self, instruction, environment, context) -> None:
        candidate_file = self.logs_dir / Path(self._candidate_path).name
        candidate_file.write_text(self._response, encoding="utf-8")
        await environment.upload_file(candidate_file, self._candidate_path)

        command = self._candidate_command.format(
            candidate_path=shlex.quote(self._candidate_path)
        )
        result = await environment.exec(command=command)
        context.metadata = {
            "candidate_path": self._candidate_path,
            "candidate_return_code": result.return_code,
            "candidate_stdout": result.stdout,
            "candidate_stderr": result.stderr,
        }


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

    raise FileNotFoundError("Harbor task files are unavailable.")


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_HARBOR_CODE_FENCE_RE = re.compile(
    r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE
)


def extract_harbor_candidate(text: str) -> str:
    text = _THINK_RE.sub("", text)
    match = _HARBOR_CODE_FENCE_RE.search(text)
    return match.group(1).strip() if match else text.strip()


def _resource_mode(policy: str):
    trial_config = importlib.import_module("harbor.models.trial.config")
    modes = {
        "reserve": trial_config.ResourceMode.GUARANTEE,
        "limit": trial_config.ResourceMode.LIMIT,
        "ignore": trial_config.ResourceMode.IGNORE,
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
    environment_type = importlib.import_module("harbor.models.environment_type")
    trial_config = importlib.import_module("harbor.models.trial.config")
    trial_mod = importlib.import_module("harbor.trial.trial")

    resolved_task_path = Path(task_path).resolve()
    if not resolved_task_path.is_dir():
        raise FileNotFoundError(f"Harbor task directory does not exist: {task_path}")

    with TemporaryDirectory(prefix="training-gym-harbor-trial-") as trials_dir:
        trial = await trial_mod.Trial.create(
            trial_config.TrialConfig(
                trials_dir=Path(trials_dir),
                task=trial_config.TaskConfig(path=resolved_task_path),
                agent=trial_config.AgentConfig(
                    import_path=_AGENT_IMPORT_PATH,
                    override_timeout_sec=float(timeout_sec),
                    kwargs={
                        "response": response,
                        "candidate_path": candidate_path,
                        "candidate_command": candidate_command,
                    },
                ),
                verifier=trial_config.VerifierConfig(
                    override_timeout_sec=float(timeout_sec)
                ),
                environment=trial_config.EnvironmentConfig(
                    type=environment_type.EnvironmentType.MODAL,
                    cpu_enforcement_policy=_resource_mode(cpu_policy),
                    memory_enforcement_policy=_resource_mode(memory_policy),
                    override_cpus=max(1, int(sandbox_cpu)),
                    override_memory_mb=int(sandbox_memory),
                    kwargs={
                        "sandbox_timeout_secs": max(300, timeout_sec * 3),
                        "sandbox_idle_timeout_secs": max(120, timeout_sec * 2),
                    },
                ),
            )
        )
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


async def score_from_label(
    response: str,
    label: str | dict[str, Any],
    **score_kwargs: Any,
) -> tuple[float, dict[str, Any]]:
    parsed = _parse_label(label)
    return await score_harbor_response(
        response,
        task_path=resolve_harbor_task_path(parsed),
        candidate_path=parsed.get(
            "harbor_candidate_path", "/tmp/training-gym-candidate.py"
        ),
        candidate_command=parsed.get(
            "harbor_candidate_command", "python {candidate_path}"
        ),
        **score_kwargs,
    )


async def harbor_reward(args, sample, **kwargs):
    if isinstance(sample, list):
        return await asyncio.gather(
            *(harbor_reward(args, item, **kwargs) for item in sample)
        )

    score, metadata = await score_from_label(
        extract_harbor_candidate(sample.response),
        sample.label,
    )
    sample.metadata = {
        **(getattr(sample, "metadata", None) or {}),
        "harbor": metadata,
    }
    return score
