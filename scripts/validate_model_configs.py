"""
Input: string [ model name ]
Output: string [ Formatted test result ]
Optional args:
    -j: json formatted output
    -o: output file path
"""

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from modal_training_gym.common.dataset import (
    DatasetConfig,
    HuggingFaceDataset,
    MultimodalDataset,
)
from modal_training_gym.common.models.qwen3_asr_1_7b import Qwen3_ASR_1_7B
from modal_training_gym.common.models.validation import VALIDATABLE_MODELS
from modal_training_gym.common.run import TrainingRun, TrainingRunStatus
from modal_training_gym.common.step_timing import measured_run_times
from modal_training_gym.common.wandb import WandbConfig
from modal_training_gym.model import ModelConfig
from modal_training_gym.train import TrainConfig
from modal_training_gym.train_recipes.slime_recipe import SlimeRecipe

VALIDATION_EPHEMERAL_DISK_MIB = 2_097_152


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


def _total_step_time_s(result: "TutorialResult") -> float:
    """Sum of per-step durations.

    Reported instead of wall clock, which also covers queue, model download and
    checkpoint conversion time — variable with compute availability rather than
    gym performance.
    """
    return float(
        sum(step.get("duration_s") or 0 for step in (result.step_times or {}).values())
    )


def _step_keys(result: "TutorialResult") -> list[str]:
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
class TutorialResult:
    base_model_name: str
    step_count: int
    training_run_id: str
    training_run_status: TrainingRunStatus
    total_duration_s: float
    step_times: dict[str, dict[str, int | None]] | None = None
    substep_times: dict[str, dict[str, dict[str, float | None]]] | None = None

    @property
    def succeeded(self) -> bool:
        return self.training_run_status == TrainingRunStatus.COMPLETED

    def print_summary(self) -> None:
        print(f"Training run result for {self.training_run_id}")
        print("Parameters:")
        print(f"Base model name: {self.base_model_name}")
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
    def from_dict(cls, data: dict) -> "TutorialResult":
        return cls(
            base_model_name=data["base_model_name"],
            step_count=data["step_count"],
            training_run_id=data["training_run_id"],
            training_run_status=TrainingRunStatus(data["training_run_status"]),
            total_duration_s=data["total_duration_s"],
            step_times=data.get("step_times"),
            substep_times=data.get("substep_times"),
        )


class Gsm8kDataset(HuggingFaceDataset):
    hf_repo = "openai/gsm8k"
    hf_config = "main"
    input_column = "question"
    output_column = "answer"
    output_format = "jsonl"
    apply_chat_template = True
    always_prepare = True

    def load(self, split: str = "all"):
        from datasets import load_dataset

        ds = load_dataset(self.hf_repo, self.hf_config, split=self.hf_split)
        if self.n_rows:
            ds = ds.select(range(min(self.n_rows, len(ds))))
        return ds.map(lambda r: {"answer": r["answer"].split("####")[-1].strip()})


class LibriSpeechASRDataset(MultimodalDataset):
    """LibriSpeech ASR rows (prompt + audio data-URI + transcript label).

    Mirrors the 006_audio_asr tutorial dataset: audio models can't train on
    gsm8k, so they validate against a handful of LibriSpeech clips instead.
    """

    modality = "audio"
    hf_repo = "hf-internal-testing/librispeech_asr_dummy"
    hf_config = "clean"
    hf_split = "validation"
    n_rows = 8
    always_prepare = True
    apply_chat_template = False

    _INSTRUCTION = (
        "<audio>\nTranscribe the speech to text. Respond with only the transcript."
    )

    def __init__(self, **kwargs):
        super().__init__(rows=[], **kwargs)

    def _build_rows(self) -> list[dict]:
        import base64 as b64
        import io

        import soundfile as sf
        from datasets import Audio, load_dataset

        ds = load_dataset(self.hf_repo, self.hf_config, split=self.hf_split)
        ds = ds.select(range(min(self.n_rows, len(ds))))
        ds = ds.cast_column("audio", Audio(decode=False))
        rows = []
        for ex in ds:
            audio = ex["audio"]
            data = (
                audio["bytes"]
                if audio.get("bytes")
                else open(audio["path"], "rb").read()
            )
            arr, sr = sf.read(io.BytesIO(data))
            buf = io.BytesIO()
            sf.write(buf, arr, sr, format="WAV")
            data_uri = "data:audio/wav;base64," + b64.b64encode(buf.getvalue()).decode(
                "ascii"
            )
            rows.append(
                {
                    self.input_key: self._INSTRUCTION,
                    self.media_column: [data_uri],
                    self.label_key: ex["text"].lower().strip(),
                }
            )
        return rows

    def load(self, split: str = "all") -> list[dict]:
        return self._build_rows()

    def prepare(self, path, eval_paths=None):
        rows = self._build_rows()
        self._write_jsonl(rows, path)
        if eval_paths:
            for eval_path in eval_paths.values():
                self._write_jsonl(rows, eval_path)


def pick_dataset(model_config: ModelConfig) -> DatasetConfig:
    """Pick a validation dataset matching the base model's modality.

    Audio models (Qwen3-ASR) need speech clips, so they get LibriSpeech;
    everything else defaults to gsm8k.
    """
    if isinstance(model_config, Qwen3_ASR_1_7B):
        return LibriSpeechASRDataset(n_rows=8)
    return Gsm8kDataset(n_rows=10)


def _model_config_registry() -> dict[str, type[ModelConfig]]:
    """Map normalized model names to their ModelConfig subclass.

    Keys cover both the full HF repo id ("qwen/qwen3-4b") and the short
    repo name ("qwen3-4b"), all lowercased.
    """
    registry: dict[str, type[ModelConfig]] = {}
    for name, model_config in VALIDATABLE_MODELS:
        registry[name.lower()] = model_config
        registry[model_config.model_name.lower()] = model_config
    return registry


def available_model_names() -> list[str]:
    """Sorted short model names (e.g. "qwen3-4b") validatable on slime.

    The shared registry excludes models with no base slime recipe (e.g. Kimi on
    miles), since this script only runs base training on slime.
    """
    return sorted(name for name, _ in VALIDATABLE_MODELS)


def get_model_config_from_model_name(model_name: str) -> ModelConfig:
    registry = _model_config_registry()
    config_cls = registry.get(model_name.lower())
    if config_cls is None:
        available = sorted({cls.model_name for cls in registry.values()})
        raise ValueError(
            f"unknown model {model_name!r}; available: {', '.join(available)}"
        )
    return config_cls()


def run_base_training_on_slime(
    model_name: str,
    step_count: int = 1,
    wandb_project: str | None = None,
    wandb_group: str | None = None,
    wandb_secret_name: str = "wandb-secret",
    eval_interval: int | None = None,
    save_interval: int | None = None,
    colocate: bool | None = None,
) -> TutorialResult:
    model_config = get_model_config_from_model_name(model_name)
    dataset = pick_dataset(model_config)
    dataset_name = getattr(dataset, "hf_repo", type(dataset).__name__).rsplit("/", 1)[
        -1
    ]
    model_short_name = model_config.model_name.rsplit("/", 1)[-1]
    train_recipe = SlimeRecipe.get_base_recipe(model_config)
    train_recipe.num_rollout = step_count
    if colocate is not None:
        train_recipe.colocate = colocate
    if colocate is False and train_recipe.rollout_num_gpus is None:
        train_recipe.rollout_num_gpus = (
            train_recipe.actor_num_nodes * train_recipe.actor_num_gpus_per_node
        )
    if eval_interval is not None:
        train_recipe.eval_interval = eval_interval
    if save_interval is not None:
        train_recipe.save_interval = save_interval
    train_recipe.rm_type = "deepscaler"
    train_recipe.train_function_kwargs = {
        **dict(train_recipe.train_function_kwargs or {}),
        "ephemeral_disk": VALIDATION_EPHEMERAL_DISK_MIB,
    }
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
    step_times, substep_times = measured_run_times(train_result.training_run_id)

    return TutorialResult(
        base_model_name=model_name,
        step_count=step_count,
        training_run_id=train_result.training_run_id,
        training_run_status=training_run.status,
        total_duration_s=float(training_run.duration_seconds or 0.0),
        step_times=step_times,
        substep_times=substep_times,
    )


def _status_label(result: TutorialResult) -> str:
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
    result: TutorialResult, baseline_path: Path, dashboard_url: str | None
) -> str:
    """Format the duration change vs a baseline result, naming the baseline
    run, e.g. "+500.0s (+33%) from [`run-id`](https://…/training/run-id)".
    """
    if not baseline_path.is_file():
        return "—"
    baseline = TutorialResult.from_dict(json.loads(baseline_path.read_text()))
    delta = (
        _format_secs_delta(
            _total_step_time_s(result),
            _total_step_time_s(baseline),
        )
        or "—"
    )
    return f"{delta} from {_training_run_link(baseline.training_run_id, dashboard_url)}"


def _load_baseline(baseline_path: Path | None) -> TutorialResult | None:
    if baseline_path is None or not baseline_path.is_file():
        return None
    return TutorialResult.from_dict(json.loads(baseline_path.read_text()))


def _load_baseline_meta(baseline_path: Path | None) -> BaselineMeta | None:
    """Load sidecar meta written by ``download_perf_baseline.py``."""
    if baseline_path is None:
        return None
    meta_path = baseline_path.with_name(baseline_path.stem + ".meta.json")
    if not meta_path.is_file():
        return None
    return BaselineMeta.from_dict(json.loads(meta_path.read_text()))


def _format_result_details(
    result: TutorialResult,
    baseline: TutorialResult | None = None,
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
        result = TutorialResult.from_dict(json.loads(path.read_text()))
        status = _status_label(result)
        row = (
            f"| {result.base_model_name} | {status} "
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

    header = "| Model | Status | Step time | Steps | Run |"
    divider = "| --- | --- | --- | --- | --- |"
    empty = "| _no results_ | | | | |"
    if baseline_dir is not None:
        header += " Delta |"
        divider += " --- |"
        empty += " |"

    lines = [
        "<!-- validate-models-comment -->",
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
        description="Validate a model config by running base training on slime."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    check_parser = subparsers.add_parser(
        "check", help="Run base training for a single model."
    )
    check_parser.add_argument(
        "-m",
        "--model",
        required=True,
        help="Base model name to run training on (e.g. qwen3-4b).",
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
        "--non-colocated",
        action="store_true",
        help="Allocate rollout GPUs separately from trainer GPUs.",
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

    subparsers.add_parser(
        "list", help="Print available model names as a JSON array and exit."
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
        print(json.dumps(available_model_names()))
        return

    if args.command == "summarize":
        print(
            summarize_results(args.results_dir, args.baseline_dir, args.dashboard_url)
        )
        return

    tutorial_result = run_base_training_on_slime(
        args.model,
        args.num_steps,
        None if args.no_wandb else args.wandb_project,
        args.wandb_group,
        args.wandb_secret_name,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        colocate=False if args.non_colocated else None,
    )
    tutorial_result.print_summary()

    if args.output:
        Path(args.output).write_text(json.dumps(tutorial_result.to_dict()))
    if args.json:
        print(json.dumps(tutorial_result.to_dict()))

    if not tutorial_result.succeeded:
        print("Training run failed")
        exit(1)
    print("Training run completed successfully")
    exit(0)


if __name__ == "__main__":
    __main__()
