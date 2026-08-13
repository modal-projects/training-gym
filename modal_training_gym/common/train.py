import dataclasses as _dc
import secrets as _secrets
import sys
import threading
import time
import warnings
from contextlib import nullcontext
from typing import Any
from typing import TypeVar
from typing import cast

from modal_training_gym.common.dataset import DatasetConfig
from modal_training_gym.common.errors import TrainingGymConfigError
from modal_training_gym.common.framework import Framework
from modal_training_gym.common.ids import create_hash
from modal_training_gym.common.models import ModelConfig
from modal_training_gym.common.checkpoint import Checkpoint
from modal_training_gym.common.run import TrainingRun
from modal_training_gym.common.status import (
    FrameworkStatus,
    MilesStatus,
    SlimeStatus,
)
from modal_training_gym.common.train_result import TrainResult
from modal_training_gym.common.modal_urls import modal_app_dashboard_url
from modal_training_gym.utils.metadata import MetadataStore, vol_put
from modal_training_gym.frameworks.miles import build_miles_app
from modal_training_gym.frameworks.slime import build_slime_app
from modal_training_gym.train_recipes.base import BaseTrainRecipe, RecipeType
from modal_training_gym.train_recipes.miles_recipe import MilesRecipe
from modal_training_gym.train_recipes.slime_recipe import SlimeRecipe
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass


_RecipeT = TypeVar("_RecipeT", bound=BaseTrainRecipe)


def _merge_recipe(base: BaseTrainRecipe, overrides: BaseTrainRecipe) -> BaseTrainRecipe:
    base_fields = {f.name: getattr(base, f.name) for f in _dc.fields(base)}

    # Fields that a recipe *subclass* declares in its own body are intentional
    # config and must override the model preset even when they equal the
    # framework base recipe's default (e.g. a long-context recipe pinning
    # context_parallel_size=1, or disabling use_kl_loss). We collect those by
    # walking the MRO from the concrete recipe down to — but not including — the
    # framework's base recipe class (the immediate subclass of BaseTrainRecipe,
    # e.g. SlimeRecipe / MilesRecipe). For a plain base recipe (no subclass
    # layer) this set is empty, so we fall back to "value differs from default"
    # — which keeps an untouched recipe from clobbering the preset with bare
    # defaults (e.g. a preset's n_samples_per_prompt=8 vs default 2).
    declared: set[str] = set()
    for cls in type(overrides).__mro__:
        if cls is BaseTrainRecipe or BaseTrainRecipe in getattr(cls, "__bases__", ()):
            break
        declared |= set(getattr(cls, "__annotations__", {}))

    for f in _dc.fields(overrides):
        if f.name not in base_fields:
            continue
        user_val = getattr(overrides, f.name)
        default_val = _field_default(f)
        if f.name in declared or default_val is _dc.MISSING or user_val != default_val:
            base_fields[f.name] = user_val
    return type(base)(**base_fields)


def _field_default(field: _dc.Field) -> Any:
    if field.default is not _dc.MISSING:
        return field.default
    if field.default_factory is not _dc.MISSING:
        return field.default_factory()
    return _dc.MISSING


def _try_validate_model_parallelism(
    recipe: BaseTrainRecipe, model: ModelConfig
) -> None:
    # Not every framework recipe implements this preflight.
    if validate := getattr(recipe, "validate_model_parallelism", None):
        validate(model)


def _resolve_recipe(
    model: ModelConfig,
    recipe: _RecipeT,
    *,
    merge_model_recipe: bool,
) -> _RecipeT:
    base_recipe = type(recipe).get_base_recipe(model) if merge_model_recipe else None
    if base_recipe is None:
        _try_validate_model_parallelism(recipe, model)
        return recipe
    resolved = cast(_RecipeT, _merge_recipe(base_recipe, recipe))
    _try_validate_model_parallelism(resolved, model)
    return resolved


def _convert_checkpoint_on_cache_miss(
    app: Any,
    *,
    training_run_id: str,
    framework_status_url: str,
    framework_status_token: str,
) -> bool:
    call_kwargs = {
        "training_run_id": training_run_id,
        "framework_status_url": framework_status_url,
        "framework_status_token": framework_status_token,
    }
    hf_path = app.resolve_checkpoint.remote(**call_kwargs)
    if hf_path is None:
        return False
    app.convert_checkpoint.remote(hf_path=hf_path, **call_kwargs)
    return True


def _warn_if_external_build_app() -> None:
    """Warn when ``_build_app()`` is called from outside the package.

    Hand-rolling ``_build_app()`` + ``app.run()`` + ``app.train.spawn()`` is a
    trap: the nested ``app.run()`` is ephemeral, so exiting the block (or
    Ctrl-C) stops the app and kills the spawned run — and ``modal run
    --detach`` does not help, since it only detaches the CLI's own entrypoint
    app. ``launch()`` / ``train()`` open the app with ``detach=True`` and
    persist the function-call id so the run can be waited on from anywhere.
    """
    try:
        caller = sys._getframe(2).f_globals.get("__name__", "")
    except ValueError:  # no such frame — never block a launch over a warning
        return
    if caller.startswith("modal_training_gym"):
        return
    warnings.warn(
        "TrainConfig._build_app() is private and does not start a detached "
        "app: spawning train() on it yourself means the run dies when the "
        "enclosing app.run() block exits or is interrupted. Use "
        "TrainConfig.launch() (returns a TrainingRun handle immediately) or "
        "TrainConfig.train() (blocks for the TrainResult) instead.",
        stacklevel=3,
    )


_STAGE_LABELS: dict[str, str] = {
    SlimeStatus.INITIALIZING.value: "Initializing",
    SlimeStatus.DOWNLOAD_MODEL.value: "Downloading model",
    SlimeStatus.CONVERT_MODEL.value: "Converting model",
    SlimeStatus.PREPARE_DATASET.value: "Preparing dataset",
    SlimeStatus.ROLLOUT_INITIALIZING.value: "Initializing rollouts",
    SlimeStatus.ROLLOUT_LOGGING.value: "Generating rollouts",
    SlimeStatus.EVAL_ROLLOUT_LOGGING.value: "Evaluating rollouts",
    SlimeStatus.COMPUTE_LOG_PROBS.value: "Computing log probs",
    SlimeStatus.OPTIMIZER_STEP.value: "Optimizer step",
    SlimeStatus.WEIGHT_SYNC.value: "Weight sync",
    SlimeStatus.OFFLOAD_ROLLOUT.value: "Offload rollout",
    SlimeStatus.OFFLOAD_TRAIN.value: "Offload train",
    SlimeStatus.CHECKPOINT_SAVE.value: "Saving checkpoint",
    SlimeStatus.TRAINING.value: "Training",
}


class _TrainStatusDisplay:
    """Terminal status helper for ``train()``.

    Prints a static banner once at the start, then a single concise status
    line each time the stage changes. We deliberately do **not** use
    ``rich.Live`` here: ``modal.enable_output()`` drives its own ANSI cursor
    moves for the spinner and "Running app..." line, which fight with Live's
    redraw and produce stacked/truncated panels.
    """

    def __init__(
        self,
        run_id: str,
        framework: str,
        model_name: str,
        dataset_name: str,
        framework_status_url: str,
        config_path: str,
    ) -> None:
        self.run_id = run_id
        self.framework = framework
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.framework_status_url = framework_status_url
        self.config_path = config_path
        self.started_at: float = time.time()
        self._console = None  # lazily constructed
        self._modal_app_url: str = ""
        # Stage dedupe is shared between the local orchestrator (which emits
        # download/convert directly) and the background poller (which emits
        # in-container phases), so identical consecutive stages print once.
        self._stage_lock = threading.Lock()
        self._last_stage: str | None = None
        self._poll_stop: threading.Event | None = None
        self._poll_thread: threading.Thread | None = None

    @staticmethod
    def _format_elapsed(seconds: float) -> str:
        seconds = max(0, int(seconds))
        h, rem = divmod(seconds, 3600)
        m, s = divmod(rem, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    def _get_console(self):
        from rich.console import Console

        if self._console is None:
            self._console = Console()
        return self._console

    def print_banner(self) -> None:
        from rich.panel import Panel
        from rich.style import Style
        from rich.table import Table
        from rich.text import Text

        body = Table.grid(padding=(0, 2))
        body.add_column(style="dim", no_wrap=True)
        body.add_column(overflow="fold")

        body.add_row("Run", Text(self.run_id, style="bold cyan"))
        if self.model_name:
            body.add_row("Model", self.model_name)
        if self.dataset_name:
            body.add_row("Dataset", self.dataset_name)
        body.add_row("Framework", self.framework)

        if self.framework_status_url:
            base = self.framework_status_url.replace(
                "/api/framework-status", ""
            ).rstrip("/")
            run_url = f"{base}/training/{self.run_id}"
            body.add_row(
                "Dashboard",
                Text(
                    run_url,
                    style=Style(color="blue", underline=True, link=run_url),
                ),
            )
        else:
            body.add_row(
                "Dashboard",
                Text(
                    f"(run `training-gym setup` to populate {self.config_path})",
                    style="yellow",
                ),
            )

        self._get_console().print(
            Panel(
                body,
                title="[bold]Training Gym[/bold]",
                title_align="left",
                border_style="cyan",
                padding=(1, 2),
            )
        )

    def set_modal_app_url(self, url: str) -> None:
        if url and url != self._modal_app_url:
            self._modal_app_url = url

            from rich.style import Style
            from rich.text import Text

            self._get_console().print(
                Text.assemble(
                    ("Modal app:", "dim"),
                    " ",
                    (url, Style(color="blue", underline=True, link=url)),
                )
            )

    def emit_stage(self, stage: str, detail: str = "") -> None:
        # Dedupe on the phase only (not the progress detail) so we print one
        # line per phase transition rather than spamming a line per rollout
        # step. The local orchestrator and the poller share this guard.
        with self._stage_lock:
            if stage == self._last_stage:
                return
            self._last_stage = stage
        label = _STAGE_LABELS.get(stage, stage or "—")
        elapsed = self._format_elapsed(time.time() - self.started_at)
        suffix = f" [dim]{detail}[/dim]" if detail else ""
        # One short scrolling line per stage transition. Cyan ▶ marker
        # makes it easy to spot in a wall of Modal log output.
        self._get_console().print(
            f"[cyan]▶[/cyan] [dim]\\[{elapsed}][/dim] "
            f"[bold]{label}[/bold]{suffix} [dim]({self.run_id})[/dim]"
        )

    @staticmethod
    def _progress_detail(run: "TrainingRun") -> str:
        progress = (run.metadata or {}).get("framework_progress")
        if not isinstance(progress, dict):
            return ""
        current = progress.get("current")
        total = progress.get("total")
        if isinstance(current, int) and isinstance(total, int) and total > 0:
            return f"({current}/{total})"
        return ""

    def start_polling(self, training_run_id: str, interval: float = 4.0) -> None:
        """Track in-container phases while ``app.train.remote()`` blocks.

        The local orchestrator hands off to the remote container and then
        blocks for the whole run; without this, the terminal ``Stage`` line
        freezes at the last locally-emitted stage. The container POSTs its
        phases to the dashboard, which persists them onto the run record — so
        we poll that record and emit each new phase to the terminal.
        """
        if self._poll_thread is not None:
            return
        stop = threading.Event()

        def _poll() -> None:
            while not stop.is_set():
                try:
                    run = TrainingRun.from_id(training_run_id)
                except Exception:
                    run = None
                if run is not None and run.framework_status is not None:
                    self.emit_stage(
                        run.framework_status.value, self._progress_detail(run)
                    )
                stop.wait(interval)

        thread = threading.Thread(
            target=_poll, name="training-gym-status-poller", daemon=True
        )
        self._poll_stop = stop
        self._poll_thread = thread
        thread.start()

    def stop_polling(self) -> None:
        if self._poll_stop is not None:
            self._poll_stop.set()
        if self._poll_thread is not None:
            self._poll_thread.join(timeout=2.0)
        self._poll_stop = None
        self._poll_thread = None


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class TrainConfig:
    """Compose dataset, model, and recipe into one training entrypoint.

    ## Fields

    dataset : DatasetConfig
        The training dataset. ``train()`` materializes it into the
        framework's ``/data`` volume before training if it isn't already
        present.
    model : ModelConfig
        The model to train. Carries model identity (``model_name``) and
        weight-download logic; weights are downloaded into the shared
        HuggingFace cache volume on first use and reused across runs.
    recipe : BaseTrainRecipe
        Framework recipe (``SlimeRecipe`` or ``MilesRecipe``). Selects the
        training framework and carries Modal infra settings (GPU type, node
        count, image) plus framework CLI flags.
    checkpoint : Checkpoint | None
        Checkpoint to resume training from. When ``None``, training starts
        from the base model weights. Default ``None``.
    merge_model_recipe : bool
        When ``True``, merges the known-model preset recipe (e.g.
        ``Qwen3_4b_Recipe``) onto recipe fields you left unset. Set
        ``False`` to run the recipe exactly as written, with no preset
        defaults. Default ``True``.
    detach : bool
        Whether the training app should outlive the local client. The Modal
        app is always started detached so a dropped connection can't kill a
        multi-hour run; ``detach`` controls what ``train()`` does when its
        wait for the result is interrupted (Ctrl-C, a crashed driver):
        ``True`` leaves the run going on Modal, ``False`` stops the app on
        the way out. ``launch()`` always leaves the run going, since it
        returns before training finishes. Default ``True``.
    group_id : str | None
        Shared sweep id. Set by ``TrainingGroup`` so every run in a sweep
        carries the same id, letting the dashboard group variants together.
        Not usually set by hand. Default ``None``.
    group_overrides : dict[str, Any] | None
        Per-variant parameter overrides applied by ``TrainingGroup``, keyed
        by dotted field path (e.g. ``{"recipe.lr": 1e-5}``). Recorded in
        run metadata so the dashboard can label each variant. Default
        ``None``.
    group_axes : list[str] | None
        Names of the swept parameter paths in a ``TrainingGroup`` grid.
        Recorded in run metadata for dashboard grouping; falls back to the
        keys of ``group_overrides`` when unset. Default ``None``.
    """

    # ── Composed configs (required) ─────────────────────────────────────────
    dataset: DatasetConfig
    model: ModelConfig
    recipe: BaseTrainRecipe
    checkpoint: Checkpoint | None = None
    # Known-model recipes are presets by default; complete recipes can opt out.
    merge_model_recipe: bool = True
    # Whether a run outlives the local client. The app itself is always started
    # detached (the CLI's ``modal run --detach`` only detaches the entrypoint,
    # not the nested ``app.run()`` the driver opens), so this only decides
    # whether an interrupted ``train()`` stops the app on its way out.
    detach: bool = True
    # Set by TrainingGroup so every run in a sweep shares one id — written into
    # the TrainingRun record so the dashboard can group variants together.
    group_id: str | None = None
    group_overrides: dict[str, Any] | None = None
    group_axes: list[str] | None = None

    def _generate_training_run_id(self) -> str:
        """Mint a new run id. ``launch()`` calls this once per invocation, so
        each launch of the same config gets its own TrainingRun record."""
        return create_hash(
            self.model.model_name,
            self.checkpoint.path if self.checkpoint is not None else "",
            f"{type(self.recipe).__name__}:{self.recipe.recipe_type.value}",
            self.dataset.dataset_id,
            self.model.model_path or "",
        )

    def _build_app(self, training_run_id: str | None = None):
        _warn_if_external_build_app()
        if training_run_id is None:
            training_run_id = self._generate_training_run_id()
        recipe_type = self.recipe.recipe_type
        if recipe_type == RecipeType.MILES:
            if not isinstance(self.recipe, MilesRecipe):
                raise TrainingGymConfigError(
                    f"Recipe type {recipe_type} requires MilesRecipe, got {type(self.recipe).__name__}"
                )
            return build_miles_app(
                training_run_id=training_run_id,
                miles=_resolve_recipe(
                    self.model,
                    cast(MilesRecipe, self.recipe),
                    merge_model_recipe=self.merge_model_recipe,
                ),
                model=self.model,
                dataset=self.dataset,
                checkpoint=self.checkpoint,
                name=training_run_id,
                group_id=self.group_id,
            )
        if recipe_type == RecipeType.SLIME:
            if not isinstance(self.recipe, SlimeRecipe):
                raise TrainingGymConfigError(
                    f"Recipe type {recipe_type} requires SlimeRecipe, got {type(self.recipe).__name__}"
                )
            combined = _resolve_recipe(
                self.model,
                cast(SlimeRecipe, self.recipe),
                merge_model_recipe=self.merge_model_recipe,
            )
            return build_slime_app(
                training_run_id=training_run_id,
                slime=combined,
                model=self.model,
                dataset=self.dataset,
                checkpoint=self.checkpoint,
                name=training_run_id,
                group_id=self.group_id,
            )
        if recipe_type == RecipeType.STITCH:
            from modal_training_gym.frameworks.stitch import build_stitch_app
            from modal_training_gym.train_recipes.stitch_recipe.recipe import (
                StitchRecipe,
            )

            if not isinstance(self.recipe, StitchRecipe):
                raise TrainingGymConfigError(
                    f"Recipe type {recipe_type} requires StitchRecipe, got {type(self.recipe).__name__}"
                )
            return build_stitch_app(
                training_run_id=training_run_id,
                recipe=cast(StitchRecipe, self.recipe),
                model=self.model,
                dataset=self.dataset,
                checkpoint=self.checkpoint,
                name=training_run_id,
                group_id=self.group_id,
            )
        raise TrainingGymConfigError(f"Unknown recipe type: {recipe_type}")

    # ── Run-record helpers ─────────────────────────────────────────────────

    @property
    def framework(self) -> Framework:
        if isinstance(self.recipe, SlimeRecipe):
            return Framework.SLIME
        if isinstance(self.recipe, MilesRecipe):
            return Framework.MILES
        if self.recipe.recipe_type == RecipeType.STITCH:
            return Framework.STITCH
        raise TrainingGymConfigError(
            f"Unknown recipe type: {type(self.recipe).__name__}"
        )

    def _initializing_status(self) -> FrameworkStatus:
        # stitch runs miles in the trainer, so it reports miles' phases too.
        if isinstance(self.recipe, MilesRecipe) or (
            self.recipe.recipe_type == RecipeType.STITCH
        ):
            return MilesStatus.INITIALIZING
        if isinstance(self.recipe, SlimeRecipe):
            return SlimeStatus.INITIALIZING
        raise TrainingGymConfigError(
            f"Unknown recipe type: {type(self.recipe).__name__}"
        )

    def _build_config_summary(self, training_run_id: str) -> dict[str, Any]:
        """Framework-specific TrainingRun.config summary."""
        model = self.model
        dataset = self.dataset
        recipe = self.recipe

        wandb = getattr(recipe, "wandb", None)
        summary: dict[str, Any] = {
            "model": {"model_name": model.model_name} if model else {},
            "wandb": (
                {
                    "project": wandb.project,
                    "entity": getattr(wandb, "entity", ""),
                    "group": wandb.group,
                    "run_id": training_run_id[:8],
                }
                if wandb
                else {}
            ),
            "dataset": {
                "hf_repo": getattr(dataset, "hf_repo", ""),
                "name": type(dataset).__name__,
            },
            "lr": getattr(recipe, "lr", None),
            "global_batch_size": getattr(recipe, "global_batch_size", None),
        }

        if isinstance(recipe, SlimeRecipe | MilesRecipe):
            from modal_training_gym.common.launcher_utils import (
                serialize_recipe_params,
            )

            combined = _resolve_recipe(
                model, recipe, merge_model_recipe=self.merge_model_recipe
            )
            summary["recipe"] = {
                # gpu_type is a launcher-only field (in _MILES_SKIP) so it is
                # absent from serialize_recipe_params for miles; the dashboard
                # cluster column reads recipe.gpu_type, so keep it here too.
                "gpu_type": getattr(combined, "gpu_type", None),
                **serialize_recipe_params(combined, dataset=dataset, model=model),
            }
        elif recipe.recipe_type == RecipeType.STITCH:
            from modal_training_gym.train_recipes.stitch_recipe.recipe import (
                StitchRecipe,
            )

            stitch = cast(StitchRecipe, recipe)
            summary["lr"] = stitch.train.lr
            summary["global_batch_size"] = stitch.train.global_batch_size
            summary["recipe"] = {
                "gpu_type": stitch.train.gpu_type,
                "actor_num_nodes": stitch.train.actor_num_nodes,
                "actor_num_gpus_per_node": stitch.train.actor_num_gpus_per_node,
                "served_checkpoint_format": stitch.train.served_checkpoint_format,
                "rollout_gpu": stitch.serve.gpu,
                "rollout_gpus_per_replica": stitch.serve.gpus_per_replica,
                "rollout_min_containers": stitch.serve.min_containers,
                "rollout_max_containers": stitch.serve.max_containers,
            }

        return summary

    def _build_run_metadata(self) -> dict[str, Any] | None:
        metadata: dict[str, Any] = {}
        if self.group_id:
            metadata["group_id"] = self.group_id
        if self.group_overrides is not None or self.group_axes is not None:
            overrides = dict(self.group_overrides or {})
            axes = list(self.group_axes or overrides)
            metadata["group_tags"] = {
                "group_id": self.group_id or "",
                "axes": axes,
                "overrides": overrides,
                "tags": [
                    {
                        "key": key,
                        "label": key.split(".")[-1].replace("_", " "),
                        "value": value,
                    }
                    for key, value in overrides.items()
                ],
            }
        return metadata or None

    def _build_status_display(
        self,
        training_run_id: str,
        framework_status_url: str,
        config_path: Any,
    ) -> "_TrainStatusDisplay":
        dataset_name = ""
        if self.dataset:
            dataset_name = (
                getattr(self.dataset, "hf_repo", "") or type(self.dataset).__name__
            )
        return _TrainStatusDisplay(
            run_id=training_run_id,
            framework=self.framework.value,
            model_name=getattr(self.model, "model_name", "") if self.model else "",
            dataset_name=dataset_name,
            framework_status_url=framework_status_url,
            config_path=str(config_path),
        )

    def _resolved_recipe_for_logging(self) -> BaseTrainRecipe:
        return _resolve_recipe(
            self.model, self.recipe, merge_model_recipe=self.merge_model_recipe
        )

    def context_plan_line(self) -> str | None:
        """One-line summary of the effective training context length and parallelism plan."""
        recipe = self._resolved_recipe_for_logging()
        max_tokens_per_gpu = getattr(recipe, "max_tokens_per_gpu", None)
        if max_tokens_per_gpu is None:
            return None

        context_parallel_size = getattr(recipe, "context_parallel_size", 1) or 1
        effective_context = max_tokens_per_gpu * context_parallel_size
        return (
            f"effective_train_context={effective_context:,} tokens "
            f"(max_tokens_per_gpu={max_tokens_per_gpu:,} x cp={context_parallel_size}; "
            f"tp={getattr(recipe, 'tensor_model_parallel_size', 'n/a')}, "
            f"pp={getattr(recipe, 'pipeline_model_parallel_size', 'n/a')}, "
            f"ep={getattr(recipe, 'expert_model_parallel_size', 'n/a')})"
        )

    def train(self, *, show_output: bool = True) -> TrainResult:
        """Build the app, run training, and return the TrainResult."""
        from modal_training_gym.common.modal_lifecycle import stop_app

        launch = self.launch(show_output=show_output, prepare_inputs=True)
        try:
            return launch.result(stop_app_on_success=True)
        except BaseException:
            if not self.detach and launch.modal_app_id:
                stop_app(launch.modal_app_id)
            raise

    def launch(
        self,
        *,
        show_output: bool = True,
        prepare_inputs: bool = False,
    ) -> TrainingRun:
        """Start training in a detached Modal app and return immediately."""
        import modal

        from modal_training_gym.common.config import (
            CONFIG_PATH,
            get_framework_status_url,
        )
        from modal_training_gym.common.status_reporter import enqueue_framework_status
        from modal_training_gym.cli.setup import ensure_dashboard_deployed

        training_run_id = self._generate_training_run_id()
        ensure_dashboard_deployed()
        framework_status_url = get_framework_status_url() or ""
        framework_status_token = _secrets.token_urlsafe(32)

        status_display = self._build_status_display(
            training_run_id, framework_status_url, CONFIG_PATH
        )
        if show_output:
            status_display.print_banner()
            if context_plan_line := self.context_plan_line():
                print(f"Training context: {context_plan_line}")

        created_at = int(time.time())
        run_record = TrainingRun(
            training_run_id=training_run_id,
            modal_app_id="",
            modal_app_url="",
            framework=self.framework,
            config=self._build_config_summary(training_run_id),
            framework_status=self._initializing_status(),
            created_at=created_at,
            started_at=created_at,
            metadata=self._build_run_metadata(),
        )
        try:
            run_record.save()
            vol_put(
                MetadataStore.FRAMEWORK_STATUS_TOKENS,
                training_run_id,
                {"token": framework_status_token},
            )
        except RuntimeError:
            framework_status_token = ""
        print(f"TrainingRun recorded: {training_run_id}")

        app = self._build_app(training_run_id)
        output_context = modal.enable_output() if show_output else nullcontext()
        with output_context:
            with app.run(detach=True):
                modal_app_id = app.app_id or ""
                modal_app_url = modal_app_dashboard_url(modal_app_id)
                if show_output:
                    status_display.set_modal_app_url(modal_app_url)

                run_record.modal_app_id = modal_app_id
                run_record.modal_app_url = modal_app_url
                try:
                    run_record.save()
                except RuntimeError:
                    pass

                def _set_status(
                    status: FrameworkStatus, *, is_active: bool = True
                ) -> None:
                    run_record.framework_status = status
                    if show_output:
                        status_display.emit_stage(status.value)
                    enqueue_framework_status(
                        training_run_id,
                        status.value,
                        token=framework_status_token,
                        is_active=is_active,
                    )

                megatron_to_hf_mode = getattr(self.recipe, "megatron_to_hf_mode", "")
                needs_conversion = megatron_to_hf_mode != "bridge"
                if prepare_inputs:
                    if isinstance(self.recipe, SlimeRecipe):
                        _set_status(SlimeStatus.DOWNLOAD_MODEL, is_active=False)
                        app.download.remote(
                            training_run_id=training_run_id,
                            framework_status_url=framework_status_url,
                            framework_status_token=framework_status_token,
                        )
                        if needs_conversion:
                            _set_status(SlimeStatus.CONVERT_MODEL, is_active=False)
                            _convert_checkpoint_on_cache_miss(
                                app,
                                training_run_id=training_run_id,
                                framework_status_url=framework_status_url,
                                framework_status_token=framework_status_token,
                            )
                    elif self.recipe.recipe_type == RecipeType.STITCH:
                        # No torch_dist conversion: the stitch trainer loads the
                        # HF masters through megatron-bridge. It does need its
                        # served baseline built, which is what a delta applies
                        # against, so that replaces the conversion step.
                        _set_status(MilesStatus.DOWNLOAD_MODEL, is_active=False)
                        app.download.remote()
                        _set_status(MilesStatus.PREPARE_DATASET, is_active=False)
                        app.prepare_dataset.remote()
                        _set_status(MilesStatus.CONVERT_MODEL, is_active=False)
                        app.prepare_checkpoints.remote()
                    elif isinstance(self.recipe, MilesRecipe) and needs_conversion:
                        _set_status(MilesStatus.DOWNLOAD_MODEL, is_active=False)
                        app.download.remote(
                            training_run_id=training_run_id,
                            framework_status_url=framework_status_url,
                            framework_status_token=framework_status_token,
                        )
                        _set_status(MilesStatus.CONVERT_MODEL, is_active=False)
                        _convert_checkpoint_on_cache_miss(
                            app,
                            training_run_id=training_run_id,
                            framework_status_url=framework_status_url,
                            framework_status_token=framework_status_token,
                        )

                function_call = app.train.spawn(
                    modal_app_id=modal_app_id,
                    modal_app_url=modal_app_url,
                    framework_status_url=framework_status_url,
                    framework_status_token=framework_status_token,
                )

        run_record.function_call_id = function_call.object_id
        run_record._function_call = function_call
        run_record._status_display = status_display if show_output else None
        try:
            run_record.save()
        except RuntimeError:
            pass
        print(
            f"Launched training {run_record.training_run_id}: "
            f"app={run_record.modal_app_id}, function_call={run_record.function_call_id}"
        )
        return run_record
