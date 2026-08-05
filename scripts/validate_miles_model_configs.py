"""Validate a miles model config by running base training on miles.

The miles counterpart of ``validate_model_configs.py``. Miles models are big
(Kimi is 16 x 8 H200), so nothing is wired into the PR matrix yet — ``list``
returns the CI set, which is currently empty, while ``check`` can run any
model in ``MILES_MODELS`` on demand.

Usage:
    uv run scripts/validate_miles_model_configs.py list
    uv run scripts/validate_miles_model_configs.py check -m Kimi-K2.5
    uv run scripts/validate_miles_model_configs.py check -m Kimi-K2.5 \
        --docker-image radixark/miles:dev-<new-tag>
    uv run scripts/validate_miles_model_configs.py summarize -d results
"""

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from modal_training_gym.common.dataset import DatasetConfig, HuggingFaceDataset
from modal_training_gym.common.errors import TrainingGymConfigError
from modal_training_gym.common.models import ModelConfig
from modal_training_gym.common.models.miles_validation import (
    MILES_MODELS,
    MILES_VALIDATABLE_MODELS,
)
from modal_training_gym.common.run import TrainingRun, TrainingRunStatus
from modal_training_gym.common.wandb import WandbConfig
from modal_training_gym.train import TrainConfig
from modal_training_gym.train_recipes.miles_recipe import MilesRecipe

COMMENT_MARKER = "<!-- validate-miles-models-comment -->"


def _fmt_secs(seconds: float | int | None) -> str:
    if seconds is None:
        return "—"
    n = float(seconds)
    if n >= 60:
        minutes = int(n // 60)
        rem = n - minutes * 60
        return f"{minutes}m {rem:.3f}s"
    return f"{n:.3f}s"


@dataclass
class MilesValidationResult:
    base_model_name: str
    recipe_name: str
    docker_image: str
    step_count: int
    training_run_id: str
    training_run_status: TrainingRunStatus
    total_duration_s: float
    error_message: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.training_run_status == TrainingRunStatus.COMPLETED

    def print_summary(self) -> None:
        print(f"Training run result for {self.training_run_id}")
        print("Parameters:")
        print(f"Base model name: {self.base_model_name}")
        print(f"Recipe: {self.recipe_name}")
        print(f"Miles image: {self.docker_image}")
        print(f"Step count: {self.step_count}")
        print("Result:")
        print(f"Training run status: {self.training_run_status}")
        print(f"Total duration (s): {self.total_duration_s}")
        if self.error_message:
            print(f"Error: {self.error_message}")

    def to_dict(self) -> dict:
        data = asdict(self)
        data["training_run_status"] = self.training_run_status.value
        data["succeeded"] = self.succeeded
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "MilesValidationResult":
        return cls(
            base_model_name=data["base_model_name"],
            recipe_name=data["recipe_name"],
            docker_image=data["docker_image"],
            step_count=data["step_count"],
            training_run_id=data["training_run_id"],
            training_run_status=TrainingRunStatus(data["training_run_status"]),
            total_duration_s=data["total_duration_s"],
            error_message=data.get("error_message"),
        )


class DapoMath17kDataset(HuggingFaceDataset):
    """DAPO-Math-17k prompts, as used by the Kimi multinode tutorials."""

    hf_repo = "zhuzilin/dapo-math-17k"
    input_column = ""
    output_column = ""
    input_key = "prompt"
    label_key = "label"
    output_format = "jsonl"
    apply_chat_template = True
    always_prepare = True


def pick_dataset(n_rows: int) -> DatasetConfig:
    """Validation dataset for miles models.

    Every miles recipe today is a math-RL recipe scored by ``deepscaler``, so
    they all validate against DAPO-Math-17k.
    """
    return DapoMath17kDataset(n_rows=n_rows)


def _model_for_name(model_name: str) -> tuple[str, ModelConfig]:
    for name, model_config_cls in MILES_MODELS:
        if model_name.lower() in (name.lower(), model_config_cls.model_name.lower()):
            return name, model_config_cls()
    available = ", ".join(name for name, _ in MILES_MODELS)
    raise ValueError(f"unknown miles model {model_name!r}; available: {available}")


def get_base_recipe(model_config: ModelConfig) -> MilesRecipe:
    """The model's base miles recipe.

    ``MilesRecipe.get_base_recipe`` returns None for a model miles has no
    recipe for; validation has nothing to run in that case.
    """
    recipe = MilesRecipe.get_base_recipe(model_config)
    if recipe is None:
        raise TrainingGymConfigError(
            f"no base miles recipe for model {model_config.model_name!r}"
        )
    return recipe


def available_model_names() -> list[str]:
    """Sorted model names validated by CI — empty until a miles model is cheap
    enough to gate PRs on (see ``MILES_VALIDATABLE_MODELS``)."""
    return sorted(name for name, _ in MILES_VALIDATABLE_MODELS)


def runnable_model_names() -> list[str]:
    """Sorted model names ``check --model`` accepts."""
    return sorted(name for name, _ in MILES_MODELS)


def run_base_training_on_miles(
    model_name: str,
    step_count: int = 1,
    docker_image: str | None = None,
    n_rows: int = 64,
    wandb_project: str | None = None,
    wandb_group: str | None = None,
    wandb_secret_name: str = "wandb-secret",
    eval_interval: int | None = None,
    save_interval: int | None = None,
) -> MilesValidationResult:
    name, model_config = _model_for_name(model_name)
    dataset = pick_dataset(n_rows)
    dataset_name = getattr(dataset, "hf_repo", type(dataset).__name__).rsplit("/", 1)[
        -1
    ]
    model_short_name = model_config.model_name.rsplit("/", 1)[-1]

    train_recipe = get_base_recipe(model_config)
    train_recipe.num_rollout = step_count
    if docker_image is not None:
        train_recipe.docker_image = docker_image
    if eval_interval is not None:
        train_recipe.eval_interval = eval_interval
    if save_interval is not None:
        train_recipe.save_interval = save_interval
    if wandb_project is not None:
        train_recipe.wandb = WandbConfig(
            project=wandb_project
            or f"miles-validation-{model_short_name}-{dataset_name}",
            group=wandb_group
            or f"miles-model-validator-{model_short_name}-{dataset_name}",
            modal_wandb_secret_name=wandb_secret_name,
        )

    train_config = TrainConfig(
        model=model_config,
        dataset=dataset,
        recipe=train_recipe,
    )

    # launch() + result() rather than train(): a failed run raises out of
    # result(), and this way the run id is already in hand, so the failure
    # still produces a result record for the summary table. Otherwise this
    # mirrors TrainConfig.train() — including tearing the app down on failure,
    # so a validation run never leaves a cluster up.
    from modal_training_gym.common.modal_lifecycle import stop_app

    launched = train_config.launch(prepare_inputs=True)
    training_run_id = launched.training_run_id
    error: Exception | None = None
    try:
        launched.result(stop_app_on_success=True)
    except Exception as exc:  # noqa: BLE001 — reported, then re-derived below
        error = exc
        print(f"Training run {training_run_id} raised: {exc}")
        if not train_config.detach and launched.modal_app_id:
            stop_app(launched.modal_app_id)

    training_run = TrainingRun.from_id(training_run_id)
    status = training_run.status
    if error is not None and status == TrainingRunStatus.COMPLETED:
        # The run record can lag a crash on the driver side; trust the raise.
        status = TrainingRunStatus.FAILED

    return MilesValidationResult(
        base_model_name=name,
        recipe_name=type(train_recipe).__name__,
        docker_image=train_recipe.docker_image,
        step_count=step_count,
        training_run_id=training_run_id,
        training_run_status=status,
        total_duration_s=float(training_run.duration_seconds or 0.0),
        error_message=str(error) if error is not None else training_run.error_message,
    )


def _status_label(result: MilesValidationResult) -> str:
    if result.succeeded:
        return "✅ completed"
    return f"❌ {result.training_run_status.value}"


def _training_run_link(training_run_id: str, dashboard_url: str | None) -> str:
    """Training run id in backticks, linked to the dashboard if a base URL is given."""
    if not dashboard_url:
        return f"`{training_run_id}`"
    base = dashboard_url.rstrip("/")
    return f"[`{training_run_id}`]({base}/training/{training_run_id})"


def summarize_results(results_dir: str, dashboard_url: str | None = None) -> str:
    """Render a markdown table from result JSON files written by ``check -o``."""
    rows = []
    failures: list[str] = []
    for path in sorted(Path(results_dir).glob("*.json")):
        result = MilesValidationResult.from_dict(json.loads(path.read_text()))
        rows.append(
            f"| {result.base_model_name} | {result.recipe_name} "
            f"| `{result.docker_image}` | {_status_label(result)} "
            f"| {_fmt_secs(result.total_duration_s)} | {result.step_count} "
            f"| {_training_run_link(result.training_run_id, dashboard_url)} |"
        )
        if not result.succeeded and result.error_message:
            failures.extend(
                [
                    "<details>",
                    f"<summary>{result.base_model_name} — failure</summary>",
                    "",
                    "```",
                    result.error_message,
                    "```",
                    "",
                    "</details>",
                    "",
                ]
            )

    lines = [
        COMMENT_MARKER,
        "## Miles Model Validation Results",
        "",
        "| Model | Recipe | Miles image | Status | Duration | Steps | Run |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    lines.extend(rows or ["| _no results_ | | | | | | |"])
    if failures:
        lines.extend(["", "### Failures", ""])
        lines.extend(failures)
    return "\n".join(lines).rstrip() + "\n"


def __main__():
    parser = argparse.ArgumentParser(
        description="Validate a model config by running base training on miles."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    check_parser = subparsers.add_parser(
        "check", help="Run base training for a single model."
    )
    check_parser.add_argument(
        "-m",
        "--model",
        required=True,
        help=f"Model to run training on. One of: {', '.join(runnable_model_names())}.",
    )
    check_parser.add_argument(
        "-n",
        "--num_steps",
        type=int,
        default=1,
        help="Number of training steps (rollouts) to run. Defaults to 1.",
    )
    check_parser.add_argument(
        "--docker-image",
        default=None,
        help="Override the recipe's miles image, e.g. to test a miles bump.",
    )
    check_parser.add_argument(
        "--n-rows",
        type=int,
        default=64,
        help="Prompt rows to materialize. Must cover one rollout batch. Defaults to 64.",
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
        help="W&B group for validator runs. Defaults to miles-model-validator-{model}-{dataset}.",
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

    subparsers.add_parser(
        "list", help="Print CI-validated model names as a JSON array and exit."
    )
    subparsers.add_parser(
        "list-runnable",
        help="Print every model `check` accepts as a JSON array and exit.",
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
        "--dashboard-url",
        help="Base URL of the training dashboard. If omitted, run ids are not linked.",
    )

    args = parser.parse_args()

    if args.command == "list":
        print(json.dumps(available_model_names()))
        return

    if args.command == "list-runnable":
        print(json.dumps(runnable_model_names()))
        return

    if args.command == "summarize":
        print(summarize_results(args.results_dir, args.dashboard_url))
        return

    result = run_base_training_on_miles(
        args.model,
        args.num_steps,
        docker_image=args.docker_image,
        n_rows=args.n_rows,
        wandb_project=None if args.no_wandb else args.wandb_project,
        wandb_group=args.wandb_group,
        wandb_secret_name=args.wandb_secret_name,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
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
