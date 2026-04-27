"""Harbor evaluation helpers for checkpoint-backed training runs.

Runs Harbor trials concurrently against a served model checkpoint and
collects structured results. Loaded on the remote image where ``harbor``
is installed.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EvalResult:
    task_name: str
    task_path: str
    reward: float
    exit_status: str
    eval_report: dict[str, Any]
    agent_metrics: dict[str, Any]
    exception_type: str | None = None
    exception_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_name": self.task_name,
            "task_path": self.task_path,
            "reward": self.reward,
            "exit_status": self.exit_status,
            "eval_report": self.eval_report,
            "agent_metrics": self.agent_metrics,
            "exception_type": self.exception_type,
            "exception_message": self.exception_message,
        }


def write_eval_artifacts(
    output_dir: Path, summary: dict[str, Any], results: list[EvalResult]
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    with (output_dir / "results.jsonl").open("w") as fh:
        for result in results:
            fh.write(json.dumps(result.to_dict(), sort_keys=True) + "\n")
    failures = [r.to_dict() for r in results if r.exit_status != "ok"]
    if failures:
        (output_dir / "failures.json").write_text(
            json.dumps(failures, indent=2, sort_keys=True) + "\n"
        )


async def _evaluate_record(
    record: dict[str, Any],
    *,
    server_url: str,
    agent_import_path: str,
    agent_model_name: str,
    agent_kwargs: dict[str, Any],
    sandbox_timeout_secs: int,
    sandbox_idle_timeout_secs: int,
    environment_import_path: str | None = None,
) -> EvalResult:
    from harbor.models.environment_type import EnvironmentType
    from harbor.models.trial.config import (
        AgentConfig,
        EnvironmentConfig,
        TaskConfig,
        TrialConfig,
    )
    from harbor.trial.trial import Trial

    from .agent_function import load_verifier_report

    metadata = dict(record.get("metadata", {}))
    task_path = str(metadata["harbor_task_path"])
    task_name = str(metadata.get("harbor_task_name", Path(task_path).name))

    merged_kwargs = dict(agent_kwargs)
    merged_kwargs.update(metadata.get("harbor_agent_kwargs", {}))
    request_kwargs = dict(merged_kwargs.pop("request_kwargs", {}))
    if request_kwargs:
        merged_kwargs["request_kwargs"] = request_kwargs

    started = time.time()
    config = TrialConfig(
        task=TaskConfig(path=Path(task_path)),
        trials_dir=Path("/tmp/harbor-evals"),
        trial_name=f"{task_name}-{int(started * 1000)}-{uuid.uuid4().hex[:8]}",
        agent=AgentConfig(
            import_path=agent_import_path,
            model_name=agent_model_name,
            kwargs={
                **merged_kwargs,
                "base_url": f"{server_url.rstrip('/')}/v1",
            },
        ),
        environment=EnvironmentConfig(
            type=EnvironmentType.MODAL,
            import_path=environment_import_path,
            force_build=False,
            delete=True,
            kwargs={
                "sandbox_timeout_secs": sandbox_timeout_secs,
                "sandbox_idle_timeout_secs": sandbox_idle_timeout_secs,
            },
        ),
    )

    trial = await Trial.create(config)
    result = await trial.run()

    fallback = result.verifier_result.rewards if result.verifier_result else {}
    reward = float(fallback.get("reward", 0.0)) if fallback else 0.0
    reward, eval_report = load_verifier_report(
        trials_dir=config.trials_dir,
        trial_name=config.trial_name,
        fallback_rewards=fallback,
    )

    agent_metrics = {
        "total_time": time.time() - started,
        "model_latency": (
            result.agent_result.metadata.get("model_latency", 0.0)
            if result.agent_result and result.agent_result.metadata
            else 0.0
        ),
        "verifier_latency": (
            (result.verifier.finished_at - result.verifier.started_at).total_seconds()
            if result.verifier
            and result.verifier.started_at
            and result.verifier.finished_at
            else 0.0
        ),
    }
    exception_info = getattr(result, "exception_info", None)
    exception_message = None
    if exception_info is not None:
        for attr_name in ("message", "exception_message"):
            raw = getattr(exception_info, attr_name, None)
            if raw:
                exception_message = str(raw)
                break
        if exception_message is None:
            raw_exception = getattr(exception_info, "exception", None)
            if raw_exception:
                exception_message = str(raw_exception)
        if exception_message is None:
            exception_message = str(exception_info)

    return EvalResult(
        task_name=task_name,
        task_path=task_path,
        reward=reward,
        exit_status=(
            "ok"
            if exception_info is None
            else str(getattr(exception_info, "exception_type", "error"))
        ),
        eval_report=eval_report,
        agent_metrics=agent_metrics,
        exception_type=str(getattr(exception_info, "exception_type", "")) or None,
        exception_message=exception_message,
    )


async def run_harbor_eval(
    *,
    records: list[dict[str, Any]],
    server_url: str,
    agent_import_path: str,
    agent_model_name: str,
    agent_kwargs: dict[str, Any],
    sandbox_timeout_secs: int = 1800,
    sandbox_idle_timeout_secs: int = 300,
    environment_import_path: str | None = None,
    max_concurrency: int = 4,
) -> tuple[dict[str, Any], list[EvalResult]]:
    """Run Harbor evaluation on a list of task records concurrently."""
    if not records:
        raise ValueError("No task records to evaluate")

    started_at = datetime.now(timezone.utc)
    completed = 0
    semaphore = asyncio.Semaphore(max(1, max_concurrency))

    async def _run_one(record: dict[str, Any]) -> EvalResult:
        nonlocal completed
        async with semaphore:
            return await _evaluate_record(
                record,
                server_url=server_url,
                agent_import_path=agent_import_path,
                agent_model_name=agent_model_name,
                agent_kwargs=agent_kwargs,
                sandbox_timeout_secs=sandbox_timeout_secs,
                sandbox_idle_timeout_secs=sandbox_idle_timeout_secs,
                environment_import_path=environment_import_path,
            )

    tasks = [asyncio.create_task(_run_one(r)) for r in records]
    results: list[EvalResult] = []
    for finished in asyncio.as_completed(tasks):
        result = await finished
        results.append(result)
        completed += 1
        print(
            f"[eval] {completed}/{len(records)} "
            f"task={result.task_name} reward={result.reward:.3f} "
            f"status={result.exit_status}"
        )

    results.sort(key=lambda r: r.task_name)
    rewards = [r.reward for r in results]
    ended_at = datetime.now(timezone.utc)
    ok_count = sum(1 for r in results if r.exit_status == "ok")

    summary = {
        "started_at": started_at.isoformat(),
        "ended_at": ended_at.isoformat(),
        "task_count": len(results),
        "ok_count": ok_count,
        "failure_count": len(results) - ok_count,
        "reward_mean": (sum(rewards) / len(rewards)) if rewards else 0.0,
        "reward_max": max(rewards) if rewards else 0.0,
        "reward_min": min(rewards) if rewards else 0.0,
    }
    return summary, results
