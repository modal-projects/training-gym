"""Validate a model config by running base training on its framework.

The model registry (``common/models/validation.py``) says which framework
trains each model and whether it is cheap enough to gate PRs on; the matching
backend in ``scripts/validation_backends/`` supplies the recipe and dataset.
Everything below is framework-agnostic.

Usage:
    uv run scripts/validate_model_configs.py list
    uv run scripts/validate_model_configs.py list --all --framework miles
    uv run scripts/validate_model_configs.py check -m qwen3-4b
    uv run scripts/validate_model_configs.py check -m Kimi-K2.5 \
        --docker-image radixark/miles:dev-<new-tag>
    uv run scripts/validate_model_configs.py summarize -d results
"""

import argparse
import json
import sys
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import cloudpickle

try:
    # Run as a script: sys.path[0] is scripts/, matching download_perf_baseline.
    from validation_backends import RecipeOverrides, backend_for
except ImportError:  # imported as scripts.validate_model_configs, e.g. by tests
    from scripts.validation_backends import RecipeOverrides, backend_for

from modal_training_gym.common.models.validation import (
    ValidationFramework,
    find_validation_target,
    validation_targets,
)
from modal_training_gym.common.run import TrainingRun, TrainingRunStatus
from modal_training_gym.common.wandb import WandbConfig
from modal_training_gym.train import TrainConfig

COMMENT_MARKER = "<!-- validate-models-comment -->"


def _fmt_secs(seconds: float | int | None) -> str:
    if seconds is None:
        return "—"
    n = float(seconds)
    if n >= 60:
        minutes = int(n // 60)
        rem = n - minutes * 60
        return f"{minutes}m {rem:.3f}s"
    return f"{n:.3f}s"


def _substep_label(name: str) -> str:
    _SUBSTEP_LABELS = {
        "evaluate_rollouts": "Eval (before)",
        "generate_rollouts": "Generate rollouts",
        "offload_rollout": "Offload rollout",
        "compute_log_probs": "Compute log probs",
        "optimizer_step": "Optimizer step",
        "checkpoint_save": "Checkpoint save",
        "offload_train": "Offload train",
        "weight_sync": "Weight sync",
        "evaluate_rollouts_end": "Eval (after)",
    }

    return _SUBSTEP_LABELS.get(name, name.replace("_", " "))


def _total_step_time_s(result: "ValidationResult") -> float:
    """Sum of per-step durations.

    Reported instead of wall clock, which also covers queue, model download and
    checkpoint conversion time — variable with compute availability rather than
    gym performance.
    """
    return float(
        sum(step.get("duration_s") or 0 for step in (result.step_times or {}).values())
    )


def _step_keys(result: "ValidationResult") -> list[str]:
    keys = set(result.step_times or {}) | set(result.substep_times or {})
    return sorted(keys, key=lambda k: int(k) if k.isdigit() else k)


def _ordered_substeps(
    subs: dict[str, dict[str, float | None]],
) -> list[tuple[str, dict[str, float | None]]]:
    return sorted(
        subs.items(),
        key=lambda item: (
            item[1].get("start") is None,
            item[1].get("start") or 0,
        ),
    )


@dataclass
class ValidationResult:
    base_model_name: str
    step_count: int
    training_run_id: str
    training_run_status: TrainingRunStatus
    total_duration_s: float
    step_times: dict[str, dict[str, int | None]] | None = None
    substep_times: dict[str, dict[str, dict[str, float | None]]] | None = None
    framework: str = ValidationFramework.SLIME.value
    recipe_name: str | None = None
    docker_image: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.training_run_status == TrainingRunStatus.COMPLETED

    def print_summary(self) -> None:
        print(f"Training run result for {self.training_run_id}")
        print("Parameters:")
        print(f"Base model name: {self.base_model_name}")
        print(f"Framework: {self.framework}")
        if self.recipe_name:
            print(f"Recipe: {self.recipe_name}")
        if self.docker_image:
            print(f"Image: {self.docker_image}")
        print(f"Step count: {self.step_count}")
        print("Result:")
        print(f"Training run status: {self.training_run_status}")
        print(f"Total step time (s): {_total_step_time_s(self)}")
        print(f"Total duration (s): {self.total_duration_s}")

        keys = _step_keys(self)
        if not keys:
            return

        print("Timings:")
        for key in keys:
            step = (self.step_times or {}).get(key, {})
            duration = step.get("duration_s")
            print(f"Step {key} ({_fmt_secs(duration)})")

            for name, entry in _ordered_substeps(
                (self.substep_times or {}).get(key, {})
            ):
                print(
                    f"    {_substep_label(name)}: {_fmt_secs(entry.get('duration_s'))}"
                )

    def to_dict(self) -> dict:
        data = asdict(self)
        data["training_run_status"] = self.training_run_status.value
        data["succeeded"] = self.succeeded
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "ValidationResult":
        """Rebuild a result, tolerating JSON written before a field existed.

        Baselines are downloaded from artifacts on already-merged PRs, so this
        reads results produced by older revisions of this script.
        """
        return cls(
            base_model_name=data["base_model_name"],
            step_count=data["step_count"],
            training_run_id=data["training_run_id"],
            training_run_status=TrainingRunStatus(data["training_run_status"]),
            total_duration_s=data["total_duration_s"],
            step_times=data.get("step_times"),
            substep_times=data.get("substep_times"),
            framework=data.get("framework", ValidationFramework.SLIME.value),
            recipe_name=data.get("recipe_name"),
            docker_image=data.get("docker_image"),
        )


def available_model_names(
    framework: ValidationFramework | None = None,
    *,
    include_non_gating: bool = False,
    accepts_override: str | None = None,
) -> list[str]:
    """Sorted model names, by default only those a PR may launch.

    Models with ``gates_prs=False`` (e.g. Kimi on 16 x 8 H200) are excluded
    unless asked for: they stay runnable by name from the CLI or a
    ``workflow_dispatch``, but must never enter a PR matrix.

    ``accepts_override`` narrows to models whose framework honors a given
    ``check`` flag, e.g. ``"docker-image"``. That is the only argument that
    loads framework backends; the per-PR path goes through ``diff_impact.py``
    and never reaches it.
    """
    targets = validation_targets(framework, gating_only=not include_non_gating)
    if accepts_override is not None:
        field_name = _override_field(accepts_override)
        targets = tuple(
            target
            for target in targets
            if field_name in backend_for(target.framework).supported_overrides
        )
    return sorted(target.name for target in targets)


def _override_field(flag: str) -> str:
    """Map a ``check`` flag spelling ("docker-image") to a RecipeOverrides field."""
    field_name = flag.strip().lstrip("-").replace("-", "_")
    known = {field.name for field in fields(RecipeOverrides)}
    if field_name not in known:
        available = ", ".join(sorted(name.replace("_", "-") for name in known))
        raise ValueError(f"unknown override {flag!r}; one of: {available}")
    return field_name


def _ship_dataset_definition(dataset) -> None:
    """Send the dataset's defining module by value, not by reference.

    ``resolve_caller_context`` only registers the module that calls ``train()``,
    which leaves a dataset class defined in a backend module pickled by name.
    Nothing under ``scripts/`` is importable inside the training image, so the
    container would fail to unpickle it during data preparation. Classes that
    ship with the package are importable remotely and stay by reference.
    """
    module = sys.modules.get(type(dataset).__module__)
    if module is None or module.__name__.startswith("modal_training_gym"):
        return
    cloudpickle.register_pickle_by_value(module)


def run_base_training(
    model_name: str,
    step_count: int = 1,
    wandb_project: str | None = None,
    wandb_group: str | None = None,
    wandb_secret_name: str = "wandb-secret",
    overrides: RecipeOverrides | None = None,
) -> ValidationResult:
    overrides = overrides or RecipeOverrides()
    target = find_validation_target(model_name)
    backend = backend_for(target.framework)
    model_config = target.model_config()

    train_recipe = backend.build_recipe(target, model_config, step_count, overrides)
    dataset = backend.pick_dataset(target, model_config, train_recipe, step_count)
    _ship_dataset_definition(dataset)

    dataset_name = getattr(dataset, "hf_repo", type(dataset).__name__).rsplit("/", 1)[
        -1
    ]
    model_short_name = model_config.model_name.rsplit("/", 1)[-1]
    if wandb_project is not None:
        train_recipe.wandb = WandbConfig(
            project=wandb_project
            or f"model-validation-{model_short_name}-{dataset_name}",
            group=wandb_group or f"model-validator-{model_short_name}-{dataset_name}",
            modal_wandb_secret_name=wandb_secret_name,
        )

    train_config = TrainConfig(
        model=model_config,
        dataset=dataset,
        recipe=train_recipe,
    )

    train_result = train_config.train()
    training_run = TrainingRun.from_id(train_result.training_run_id)

    return ValidationResult(
        base_model_name=target.name,
        step_count=step_count,
        training_run_id=train_result.training_run_id,
        training_run_status=training_run.status,
        total_duration_s=float(training_run.duration_seconds or 0.0),
        step_times=training_run.step_times,
        substep_times=training_run.substep_times,
        framework=target.framework.value,
        recipe_name=type(train_recipe).__name__,
        docker_image=backend.docker_image(train_recipe),
    )


def _status_label(result: ValidationResult) -> str:
    if result.succeeded:
        return "✅ completed"
    return f"❌ {result.training_run_status.value}"


def _format_secs_delta(
    current: float | int | None, baseline: float | int | None
) -> str | None:
    """Compact delta vs baseline, or None when either timing is missing."""
    if current is None or baseline is None:
        return None
    current_f = float(current)
    baseline_f = float(baseline)
    delta_s = current_f - baseline_f
    if baseline_f <= 0:
        return f"{delta_s:+.3f}s"
    percent = delta_s / baseline_f * 100
    return f"{delta_s:+.3f}s ({percent:+.0f}%)"


def _training_run_link(training_run_id: str, dashboard_url: str | None) -> str:
    """Training run id in backticks, linked to the dashboard if a base URL is given."""
    if not dashboard_url:
        return f"`{training_run_id}`"
    base = dashboard_url.rstrip("/")
    return f"[`{training_run_id}`]({base}/training/{training_run_id})"


@dataclass
class BaselineMeta:
    commit_sha: str
    commit_url: str

    @classmethod
    def from_dict(cls, data: dict) -> "BaselineMeta | None":
        sha = data.get("commit_sha")
        url = data.get("commit_url")
        if not sha or not url:
            return None
        return cls(commit_sha=str(sha), commit_url=str(url))

    def commit_link(self) -> str:
        short = self.commit_sha[:7]
        return f"[`{short}`]({self.commit_url})"


def _format_duration_delta(
    result: ValidationResult, baseline_path: Path, dashboard_url: str | None
) -> str:
    """Format the duration change vs a baseline result, naming the baseline
    run, e.g. "+500.0s (+33%) from [`run-id`](https://…/training/run-id)".
    """
    if not baseline_path.is_file():
        return "—"
    baseline = ValidationResult.from_dict(json.loads(baseline_path.read_text()))
    delta = (
        _format_secs_delta(
            _total_step_time_s(result),
            _total_step_time_s(baseline),
        )
        or "—"
    )
    return f"{delta} from {_training_run_link(baseline.training_run_id, dashboard_url)}"


def _load_baseline(baseline_path: Path | None) -> ValidationResult | None:
    if baseline_path is None or not baseline_path.is_file():
        return None
    return ValidationResult.from_dict(json.loads(baseline_path.read_text()))


def _load_baseline_meta(baseline_path: Path | None) -> BaselineMeta | None:
    """Load sidecar meta written by ``download_perf_baseline.py``."""
    if baseline_path is None:
        return None
    meta_path = baseline_path.with_name(baseline_path.stem + ".meta.json")
    if not meta_path.is_file():
        return None
    return BaselineMeta.from_dict(json.loads(meta_path.read_text()))


def _format_result_details(
    result: ValidationResult,
    baseline: ValidationResult | None = None,
    baseline_meta: BaselineMeta | None = None,
    dashboard_url: str | None = None,
) -> list[str]:
    """Markdown <details> block with run status and a consolidated timing table."""
    lines = [
        "<details>",
        f"<summary>{result.base_model_name}</summary>",
        "",
        f"{_training_run_link(result.training_run_id, dashboard_url)} — {_status_label(result)}",
    ]
    recipe_bits = [f"Framework: {result.framework}"]
    if result.recipe_name:
        recipe_bits.append(f"Recipe: `{result.recipe_name}`")
    if result.docker_image:
        recipe_bits.append(f"Image: `{result.docker_image}`")
    lines.append(" · ".join(recipe_bits))
    if baseline is not None:
        baseline_bits = [
            _training_run_link(baseline.training_run_id, dashboard_url),
        ]
        if baseline_meta is not None:
            baseline_bits.append(f"on {baseline_meta.commit_link()}")
        lines.append(f"Baseline: {' '.join(baseline_bits)}")
    lines.append("")

    keys = _step_keys(result)
    if not keys:
        lines.extend(["_No step timing data._", "", "</details>", ""])
        return lines

    if baseline is not None:
        lines.extend(
            [
                "| Phase | Duration | Delta |",
                "| --- | --- | --- |",
            ]
        )
    else:
        lines.extend(
            [
                "| Phase | Duration |",
                "| --- | --- |",
            ]
        )

    def _row(phase: str, duration: float | int | None, base: float | int | None) -> str:
        if baseline is None:
            return f"| {phase} | {_fmt_secs(duration)} |"
        delta = _format_secs_delta(duration, base) or "—"
        return f"| {phase} | {_fmt_secs(duration)} | {delta} |"

    for key in keys:
        step = (result.step_times or {}).get(key) or {}
        baseline_step = ((baseline.step_times or {}).get(key) or {}) if baseline else {}
        baseline_subs = (
            ((baseline.substep_times or {}).get(key) or {}) if baseline else {}
        )
        for name, entry in _ordered_substeps(
            (result.substep_times or {}).get(key) or {}
        ):
            base_entry = baseline_subs.get(name) or {}
            lines.append(
                _row(
                    _substep_label(name),
                    entry.get("duration_s"),
                    base_entry.get("duration_s"),
                )
            )
        lines.append(
            _row(
                f"Step {key}",
                step.get("duration_s"),
                baseline_step.get("duration_s"),
            )
        )
    if len(keys) > 1:
        lines.append(
            _row(
                "Total step time",
                _total_step_time_s(result),
                _total_step_time_s(baseline) if baseline else None,
            )
        )
    lines.extend(["", "</details>", ""])
    return lines


def summarize_results(
    results_dir: str, baseline_dir: str | None, dashboard_url: str | None = None
) -> str:
    rows = []
    details: list[str] = []
    for path in sorted(Path(results_dir).glob("*.json")):
        result = ValidationResult.from_dict(json.loads(path.read_text()))
        status = _status_label(result)
        row = (
            f"| {result.base_model_name} | {result.framework} | {status} "
            f"| {_total_step_time_s(result):.1f}s | {result.step_count} "
            f"| {_training_run_link(result.training_run_id, dashboard_url)} |"
        )
        baseline_path = (
            Path(baseline_dir) / path.name if baseline_dir is not None else None
        )
        if baseline_dir is not None:
            assert baseline_path is not None
            delta = _format_duration_delta(result, baseline_path, dashboard_url)
            row += f" {delta} |"
        rows.append(row)
        details.extend(
            _format_result_details(
                result,
                _load_baseline(baseline_path),
                _load_baseline_meta(baseline_path),
                dashboard_url,
            )
        )

    header = "| Model | Framework | Status | Step time | Steps | Run |"
    divider = "| --- | --- | --- | --- | --- | --- |"
    empty = "| _no results_ | | | | | |"
    if baseline_dir is not None:
        header += " Delta |"
        divider += " --- |"
        empty += " |"

    lines = [
        COMMENT_MARKER,
        "## Model Validation Results",
        "",
        header,
        divider,
    ]
    lines.extend(rows or [empty])
    if details:
        lines.extend(["", "### Step timings", ""])
        lines.extend(details)
    return "\n".join(lines).rstrip() + "\n"


def __main__():
    parser = argparse.ArgumentParser(
        description="Validate a model config by running base training on its framework."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    check_parser = subparsers.add_parser(
        "check", help="Run base training for a single model."
    )
    check_parser.add_argument(
        "-m",
        "--model",
        required=True,
        help="Base model name to run training on (e.g. qwen3-4b). One of: "
        f"{', '.join(available_model_names(include_non_gating=True))}.",
    )
    check_parser.add_argument(
        "-n",
        "--num_steps",
        type=int,
        default=1,
        help="Number of training steps (rollouts) to run. Defaults to 1.",
    )
    check_parser.add_argument(
        "--eval-interval",
        type=int,
        default=None,
        help="Override the recipe eval_interval (eval every N rollouts).",
    )
    check_parser.add_argument(
        "--save-interval",
        type=int,
        default=None,
        help="Override the recipe save_interval (checkpoint every N rollouts).",
    )
    check_parser.add_argument(
        "--docker-image",
        default=None,
        help="Override the recipe's image, e.g. to test a miles bump. "
        "Only frameworks that pin an image accept this (miles).",
    )
    check_parser.add_argument(
        "--non-colocated",
        action="store_true",
        help="Allocate rollout GPUs separately from trainer GPUs (slime only).",
    )
    check_parser.add_argument(
        "-o",
        "--output",
        help="Write the result as JSON to this file path.",
    )
    check_parser.add_argument(
        "-j",
        "--json",
        action="store_true",
        help="Print the result as JSON to stdout.",
    )
    check_parser.add_argument(
        "--wandb-project",
        default=None,
        help="W&B project for validator runs. If omitted, W&B logging is disabled.",
    )
    check_parser.add_argument(
        "--wandb-group",
        default="",
        help="W&B group for validator runs. Defaults to model-validator-{model}-{dataset}.",
    )
    check_parser.add_argument(
        "--wandb-secret-name",
        default="wandb-secret",
        help="Modal Secret name containing WANDB_API_KEY.",
    )
    check_parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable W&B logging for this validator run.",
    )

    list_parser = subparsers.add_parser(
        "list", help="Print available model names as a JSON array and exit."
    )
    list_parser.add_argument(
        "--framework",
        choices=[framework.value for framework in ValidationFramework],
        default=None,
        help="Only list models validated on this framework.",
    )
    list_parser.add_argument(
        "--all",
        action="store_true",
        help="Include models a PR may not launch (gates_prs=False), such as "
        "Kimi on miles. Never use this to build a PR matrix.",
    )
    list_parser.add_argument(
        "--accepts-override",
        default=None,
        metavar="FLAG",
        help="Only list models whose framework honors this `check` flag, "
        "e.g. --accepts-override docker-image.",
    )

    summarize_parser = subparsers.add_parser(
        "summarize",
        help="Render a markdown table from a directory of result JSON files.",
    )
    summarize_parser.add_argument(
        "-d",
        "--results-dir",
        required=True,
        help="Directory containing result JSON files written by `check --output`.",
    )
    summarize_parser.add_argument(
        "-b",
        "--baseline-dir",
        help="Directory containing baseline result JSON files to compare against",
    )
    summarize_parser.add_argument(
        "--dashboard-url",
        help="Base URL of the training dashboard. If omitted, run ids are not linked.",
    )

    args = parser.parse_args()

    if args.command == "list":
        print(
            json.dumps(
                available_model_names(
                    ValidationFramework(args.framework) if args.framework else None,
                    include_non_gating=args.all,
                    accepts_override=args.accepts_override,
                )
            )
        )
        return

    if args.command == "summarize":
        print(
            summarize_results(args.results_dir, args.baseline_dir, args.dashboard_url)
        )
        return

    result = run_base_training(
        args.model,
        args.num_steps,
        None if args.no_wandb else args.wandb_project,
        args.wandb_group,
        args.wandb_secret_name,
        overrides=RecipeOverrides(
            eval_interval=args.eval_interval,
            save_interval=args.save_interval,
            docker_image=args.docker_image,
            non_colocated=args.non_colocated,
        ),
    )
    result.print_summary()

    if args.output:
        Path(args.output).write_text(json.dumps(result.to_dict()))
    if args.json:
        print(json.dumps(result.to_dict()))

    if not result.succeeded:
        print("Training run failed")
        exit(1)
    print("Training run completed successfully")
    exit(0)


if __name__ == "__main__":
    __main__()
