"""Miles custom agent hook that runs Harbor trials.

This module is loaded on the remote Miles image at runtime via
``--custom-agent-function-path``. Miles calls ``run()`` for each rollout
sample; it creates a Harbor sandbox, runs the user-supplied agent, and
returns the reward + metrics.

Not importable locally — requires ``harbor`` (installed on the remote image).
"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit


def _resolve_agent_base_url(
    *, tracer_base_url: str, public_base_url: str | None
) -> str:
    """Swap the internal cluster host with the public tunnel URL, keeping the path."""
    tracer_base_url = tracer_base_url.rstrip("/")
    public_base_url = (public_base_url or "").strip().rstrip("/")
    if not public_base_url:
        return f"{tracer_base_url}/v1"

    tracer_split = urlsplit(tracer_base_url)
    public_split = urlsplit(public_base_url)
    if not tracer_split.path:
        return f"{public_base_url}/v1"

    return urlunsplit(
        (
            public_split.scheme,
            public_split.netloc,
            f"{tracer_split.path}/v1",
            public_split.query,
            public_split.fragment,
        )
    )


def _load_json_env(name: str) -> dict[str, Any]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return {}
    return json.loads(raw)


def load_verifier_report(
    trials_dir: Path,
    trial_name: str,
    fallback_rewards: dict[str, Any] | None = None,
) -> tuple[float, dict[str, Any]]:
    """Load structured verifier report from a completed Harbor trial."""
    reward = 0.0
    report: dict[str, Any] = dict(fallback_rewards or {})
    if report:
        try:
            reward = float(report.get("reward", 0.0))
        except (TypeError, ValueError):
            reward = 0.0

    report_path = trials_dir / trial_name / "verifier" / "report.json"
    if not report_path.exists():
        return reward, report

    try:
        structured = json.loads(report_path.read_text())
    except (OSError, json.JSONDecodeError):
        return reward, report

    if not isinstance(structured, dict):
        return reward, report

    try:
        reward = float(structured.get("reward", reward))
    except (TypeError, ValueError):
        pass
    return reward, structured


async def run(
    base_url: str,
    prompt: Any,
    request_kwargs: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    **kwargs,
) -> dict[str, Any] | None:
    """Harbor agent function called by Miles during rollout.

    Environment variables (set by the launcher):
      HARBOR_AGENT_IMPORT_PATH  — import path for the Harbor agent class
      HARBOR_AGENT_KWARGS       — JSON-encoded default agent kwargs
      HARBOR_PUBLIC_BASE_URL    — public tunnel URL for sandbox→model communication
      HARBOR_ENVIRONMENT_IMPORT_PATH — optional custom Harbor environment
      HARBOR_SANDBOX_TIMEOUT_SECS / HARBOR_SANDBOX_IDLE_TIMEOUT_SECS
      AGENT_MODEL_NAME          — served model name
    """
    from harbor.models.environment_type import EnvironmentType
    from harbor.models.trial.config import (
        AgentConfig,
        EnvironmentConfig,
        TaskConfig,
        TrialConfig,
    )
    from harbor.trial.trial import Trial

    metadata = metadata or {}
    request_kwargs = request_kwargs or {}
    prompt_text = (
        prompt if isinstance(prompt, str) else json.dumps(prompt, sort_keys=True)
    )

    task_path = metadata.get("harbor_task_path")
    if not task_path:
        return {
            "reward": 0.0,
            "exit_status": "missing-task-path",
            "agent_metrics": {},
            "observability_prompt": prompt_text,
        }

    trial_name = metadata.get("harbor_task_name", Path(task_path).name)

    agent_import_path = os.getenv("HARBOR_AGENT_IMPORT_PATH", "")
    if not agent_import_path:
        return {
            "reward": 0.0,
            "exit_status": "missing-agent-import-path",
            "agent_metrics": {},
            "observability_prompt": prompt_text,
        }

    agent_model_name = os.getenv("AGENT_MODEL_NAME", "model")
    agent_kwargs = _load_json_env("HARBOR_AGENT_KWARGS")
    agent_kwargs.update(metadata.get("harbor_agent_kwargs", {}))
    merged_request_kwargs = dict(agent_kwargs.pop("request_kwargs", {}))
    merged_request_kwargs.update(request_kwargs)
    if merged_request_kwargs:
        agent_kwargs["request_kwargs"] = merged_request_kwargs

    env_import_path = (
        metadata.get("harbor_environment_import_path")
        or os.getenv("HARBOR_ENVIRONMENT_IMPORT_PATH")
        or None
    )
    sandbox_timeout = int(os.getenv("HARBOR_SANDBOX_TIMEOUT_SECS", "1800"))
    sandbox_idle_timeout = int(os.getenv("HARBOR_SANDBOX_IDLE_TIMEOUT_SECS", "300"))

    resolved_base_url = _resolve_agent_base_url(
        tracer_base_url=base_url,
        public_base_url=os.getenv("HARBOR_PUBLIC_BASE_URL"),
    )

    started = time.time()
    config = TrialConfig(
        task=TaskConfig(path=Path(task_path)),
        trials_dir=Path("/tmp/harbor-trials"),
        trial_name=f"{trial_name[:40]}-{int(started * 1000)}-{uuid.uuid4().hex[:8]}",
        agent=AgentConfig(
            import_path=agent_import_path,
            model_name=agent_model_name,
            kwargs={
                **agent_kwargs,
                "base_url": resolved_base_url,
            },
        ),
        environment=EnvironmentConfig(
            type=EnvironmentType.MODAL,
            import_path=env_import_path,
            force_build=False,
            delete=True,
            kwargs={
                "sandbox_timeout_secs": sandbox_timeout,
                "sandbox_idle_timeout_secs": sandbox_idle_timeout,
            },
        ),
    )

    trial = await Trial.create(config)
    result = await trial.run()

    reward = 0.0
    fallback = result.verifier_result.rewards if result.verifier_result else {}
    if fallback:
        reward = float(fallback.get("reward", 0.0))

    reward, eval_report = load_verifier_report(
        trials_dir=config.trials_dir,
        trial_name=config.trial_name,
        fallback_rewards=fallback,
    )

    agent_metadata = (
        dict(result.agent_result.metadata)
        if result.agent_result and result.agent_result.metadata
        else {}
    )

    agent_metrics = {
        "total_time": time.time() - started,
        "model_latency": agent_metadata.get("model_latency", 0.0),
        "verifier_latency": (
            (result.verifier.finished_at - result.verifier.started_at).total_seconds()
            if result.verifier
            and result.verifier.started_at
            and result.verifier.finished_at
            else 0.0
        ),
    }
    if agent_metadata:
        agent_metrics["agent_metadata"] = agent_metadata

    return {
        "reward": reward,
        "exit_status": (
            "ok"
            if result.exception_info is None
            else result.exception_info.exception_type
        ),
        "eval_report": eval_report,
        "agent_metrics": agent_metrics,
        "observability_prompt": prompt_text,
        "observability_task_name": trial_name,
        "observability_task_path": str(task_path),
        "observability_agent_name": agent_import_path,
        "observability_response_text": (
            agent_metadata.get("raw_response_text")
            or agent_metadata.get("written_content")
            or agent_metadata.get("raw_response_preview")
        ),
        "observability_written_content": (
            agent_metadata.get("written_content")
            or agent_metadata.get("written_content_preview")
        ),
        "trajectory_messages": agent_metadata.get("trajectory_messages", []),
    }
