"""Weekly synthetic monitoring for training-gym model validation.

Test it with:

```shell
    uv run modal run -m synthetic_monitoring.model_validation --model qwen3-4b --dryrun
```
"""

from __future__ import annotations

import os
import traceback
from pathlib import Path

import modal

from modal_training_gym.common.models.validation import _ValidationConfig
from scripts.validate_model_configs import ValidationResult, run_base_training
from synthetic_monitoring.chart import (
    RunPoint,
    append_history,
    render_timing_history_chart,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
PROBE_TIMEOUT_S = 60 * 60
CLEANUP_GRACE_S = 5 * 60
LAUNCH_TIMEOUT_S = (
    len(_ValidationConfig.select(pr_only=True)) * (PROBE_TIMEOUT_S + CLEANUP_GRACE_S)
    + 30 * 60
)
MODAL_ENV = "training-gym"
SLACK_CHANNEL_ID = "C0B9ZEA3ASD"

probe_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync(uv_project_dir=str(REPO_ROOT), extra_options="--no-dev")
    .uv_pip_install("slack-sdk==3.27.1", "matplotlib==3.10.1")
    .env({"MODAL_ENVIRONMENT": MODAL_ENV})
    .add_local_python_source("modal_training_gym", "synthetic_monitoring", "scripts")
)

slack_secret = modal.Secret.from_name("gym-bot-slack", environment_name=MODAL_ENV)
modal_creds_secret = modal.Secret.from_name(
    "_training-gym-modal-creds", environment_name=MODAL_ENV
)

app = modal.App("gym-synmon-launcher")


def _slack_client():
    from slack_sdk import WebClient
    from slack_sdk.errors import SlackApiError

    client = WebClient(token=os.environ["SLACK_BOT_TOKEN"])
    try:
        client.conversations_join(channel=SLACK_CHANNEL_ID)
    except SlackApiError as exc:
        if exc.response["error"] != "already_in_channel":
            raise
    return client


@app.function(
    image=probe_image,
    timeout=PROBE_TIMEOUT_S + CLEANUP_GRACE_S,
    secrets=[slack_secret, modal_creds_secret],
)
def monitor(
    model: str = "",
    dryrun: bool = False,
    num_steps: int = 1,
) -> dict:
    selected = _ValidationConfig.find(model).name
    print(f"synmon: model={selected!r} dryrun={dryrun} num_steps={num_steps}")
    if dryrun:
        print(f"dryrun: would probe {selected!r} num_steps={num_steps}")
        return {"model": selected, "dryrun": True}

    result: ValidationResult | None = None
    try:
        result = run_base_training(selected, step_count=num_steps)
        if not result.succeeded:
            raise RuntimeError(
                f"model validation failed for {selected}: "
                f"{result.training_run_status.value}"
            )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        lines = [f"`{selected}` failed", f"```{error[:1500]}```"]
        if result:
            lines.insert(1, f"Run: `{result.training_run_id}`")
        _slack_client().chat_postMessage(
            channel=SLACK_CHANNEL_ID,
            text="\n".join(lines),
            mrkdwn=True,
            unfurl_links=False,
            unfurl_media=False,
        )
        raise

    history = append_history(
        selected, RunPoint.from_validation_result(result), environment_name=MODAL_ENV
    )
    _slack_client().files_upload_v2(
        channel=SLACK_CHANNEL_ID,
        content=render_timing_history_chart(history, model_name=selected),
        filename=f"timing_history_{selected.replace('/', '_')}.png",
        title=f"{selected} (n={len(history)} runs)",
    )

    return {
        "model": selected,
        "training_run_id": result.training_run_id,
        "total_duration_s": result.total_duration_s,
    }


@app.function(
    image=probe_image,
    timeout=LAUNCH_TIMEOUT_S,
    schedule=modal.Cron("17 0 * * 0"),
    secrets=[slack_secret],
)
def launch_weekly(
    model: str = "",
    dryrun: bool = False,
    num_steps: int = 1,
) -> list[dict]:
    names = (
        [_ValidationConfig.find(model).name]
        if model
        else [config.name for config in _ValidationConfig.select(pr_only=True)]
    )
    if not names:
        raise RuntimeError("no validatable models registered")

    results: list[dict] = []
    failures: list[str] = []
    for name, row in zip(
        names,
        monitor.map(
            names,
            kwargs={"dryrun": dryrun, "num_steps": num_steps},
            return_exceptions=True,
        ),
    ):
        if isinstance(row, Exception):
            tb = "".join(traceback.format_exception(type(row), row, row.__traceback__))
            print(f"error: probe failed for {name}:\n{tb}")
            failures.append(f"`{name}` {type(row).__name__}: {row}")
            continue
        results.append(row)

    if failures:
        summary = "\n".join(failures)
        _slack_client().chat_postMessage(
            channel=SLACK_CHANNEL_ID,
            text=f"weekly model validation had failures\n{summary}"[:3500],
            mrkdwn=True,
            unfurl_links=False,
            unfurl_media=False,
        )
        raise RuntimeError(f"synmon fan-out finished with failures:\n{summary}")
    return results


@app.local_entrypoint()
def main(
    model: str = "",
    dryrun: bool = True,
    num_steps: int = 1,
) -> None:
    result = monitor.remote(model=model, dryrun=dryrun, num_steps=num_steps)
    print(result)
