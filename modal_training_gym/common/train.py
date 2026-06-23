import dataclasses as _dc
import secrets as _secrets
import threading
import time
from enum import Enum
from typing import Any
from typing import cast

from modal_training_gym.common.dataset import DatasetConfig
from modal_training_gym.common.framework import Framework
from modal_training_gym.common.ids import create_hash
from modal_training_gym.common.models import ModelConfig
from modal_training_gym.common.checkpoint import Checkpoint
from modal_training_gym.common.run import TrainingRun, TrainingRunStatus
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
from modal_training_gym.train_recipes.miles_recipe import MilesConfig
from modal_training_gym.train_recipes.slime_recipe import SlimeRecipe
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass


def _stop_app(app_id: str) -> None:
    """Stop a detached Modal app (best effort).

    Detached apps survive client disconnect, so they don't auto-stop when the
    ``app.run()`` context exits on normal completion — we stop them explicitly.
    Mirrors what ``modal app stop`` does. Never raises.
    """
    try:
        import modal
        from modal_proto import api_pb2

        with modal.Client.from_env() as client:
            client.stub.AppStop(
                api_pb2.AppStopRequest(
                    app_id=app_id,
                    source=api_pb2.APP_STOP_SOURCE_PYTHON_CLIENT,
                )
            )
    except Exception as exc:  # noqa: BLE001 — auto-stop is best-effort
        print(f"WARNING: could not auto-stop app {app_id}: {exc}")


def _app_is_running(app_id: str) -> bool:
    """Best-effort check whether a Modal app is still alive on the server.

    A detached app outlives the client, so a dropped client-side RPC doesn't
    mean the run died — the app keeps running and lands in
    ``DETACHED_DISCONNECTED``, which we treat as live. Returns False on any
    error so callers fall back to their normal failure handling.
    """
    try:
        import modal
        from modal_proto import api_pb2

        live_states = {
            api_pb2.APP_STATE_EPHEMERAL,
            api_pb2.APP_STATE_DETACHED,
            api_pb2.APP_STATE_DETACHED_DISCONNECTED,
            api_pb2.APP_STATE_INITIALIZING,
            api_pb2.APP_STATE_DEPLOYED,
            api_pb2.APP_STATE_DERIVED,
        }
        with modal.Client.from_env() as client:
            resp = client.stub.AppGetLifecycle(
                api_pb2.AppGetLifecycleRequest(app_id=app_id)
            )
        return resp.lifecycle.app_state in live_states
    except Exception:
        return False


def _merge_recipe(base: SlimeRecipe, overrides: SlimeRecipe) -> SlimeRecipe:
    base_fields = {f.name: getattr(base, f.name) for f in _dc.fields(base)}

    # Fields that a recipe *subclass* declares in its own body are intentional
    # config and must override the model preset even when they equal the
    # framework base recipe's default (e.g. a long-context recipe pinning
    # context_parallel_size=1, or disabling use_kl_loss). We collect those by
    # walking the MRO from the concrete recipe down to — but not including — the
    # framework's base recipe class (the immediate subclass of BaseTrainRecipe,
    # e.g. SlimeRecipe / MilesConfig). For a plain base recipe (no subclass
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


class TrainStepStatus(Enum):
    INITIALIZING = "initializing"
    DOWNLOAD_MODEL = "download_model"
    CONVERT_MODEL = "convert_model"
    PREPARE_DATASET = "prepare_dataset"
    ROLLOUT_INITIALIZING = "initialize_rollouts"
    ROLLOUT_LOGGING = "generate_rollouts"
    EVAL_ROLLOUT_LOGGING = "evaluate_rollouts"
    COMPUTE_LOG_PROBS = "compute_log_probs"
    OPTIMIZER_STEP = "optimizer_step"
    WEIGHT_SYNC = "weight_sync"
    OFFLOAD_ROLLOUT = "offload_rollout"
    OFFLOAD_TRAIN = "offload_train"
    CHECKPOINT_SAVE = "checkpoint_save"
    TRAINING = "training"


_STAGE_LABELS: dict[str, str] = {
    TrainStepStatus.INITIALIZING.value: "Initializing",
    TrainStepStatus.DOWNLOAD_MODEL.value: "Downloading model",
    TrainStepStatus.CONVERT_MODEL.value: "Converting model",
    TrainStepStatus.PREPARE_DATASET.value: "Preparing dataset",
    TrainStepStatus.ROLLOUT_INITIALIZING.value: "Initializing rollouts",
    TrainStepStatus.ROLLOUT_LOGGING.value: "Generating rollouts",
    TrainStepStatus.EVAL_ROLLOUT_LOGGING.value: "Evaluating rollouts",
    TrainStepStatus.COMPUTE_LOG_PROBS.value: "Computing log probs",
    TrainStepStatus.OPTIMIZER_STEP.value: "Optimizer step",
    TrainStepStatus.WEIGHT_SYNC.value: "Weight sync",
    TrainStepStatus.OFFLOAD_ROLLOUT.value: "Offload rollout",
    TrainStepStatus.OFFLOAD_TRAIN.value: "Offload train",
    TrainStepStatus.CHECKPOINT_SAVE.value: "Saving checkpoint",
    TrainStepStatus.TRAINING.value: "Training",
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
            body.add_row(
                "Dashboard",
                Text(
                    f"{base}/training/{self.run_id}",
                    style="underline blue",
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
            self._get_console().print(
                f"[dim]Modal app:[/dim] [blue underline]{url}[/blue underline]"
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
    """Compose dataset, model, and recipe into one training entrypoint."""

    # ── Composed configs (required) ─────────────────────────────────────────
    dataset: DatasetConfig
    model: ModelConfig
    recipe: BaseTrainRecipe
    checkpoint: Checkpoint | None = None
    # Run the training app detached so it keeps running on Modal even if the
    # local client disconnects (terminal closed, laptop asleep). The CLI's
    # ``modal run --detach`` only detaches the entrypoint, not the nested
    # ``app.run()`` the driver opens — so we detach it here. Set False for an
    # attached run that Ctrl-C stops.
    detach: bool = True
    _stable_id: str | None = _dc.field(default=None, init=False, repr=False)

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def training_run_id(self) -> str:
        # Maintain same stable id, cannot change across calls on one TrainConfig.
        if self._stable_id is None:
            self._stable_id = create_hash(
                self.model.model_name,
                self.checkpoint.path if self.checkpoint is not None else "",
                f"{type(self.recipe).__name__}:{self.recipe.recipe_type.value}",
                self.dataset.dataset_id,
                self.model.model_path or "",
            )
        return self._stable_id

    def _build_app(self):
        recipe_type = self.recipe.recipe_type
        if recipe_type == RecipeType.MILES:
            if not isinstance(self.recipe, MilesConfig):
                raise TypeError(
                    f"Recipe type {recipe_type} requires MilesConfig, got {type(self.recipe).__name__}"
                )
            return build_miles_app(
                training_run_id=self.training_run_id,
                miles=cast(MilesConfig, self.recipe),
                model=self.model,
                dataset=self.dataset,
                checkpoint=self.checkpoint,
                name=self.training_run_id,
            )
        if recipe_type == RecipeType.SLIME:
            if not isinstance(self.recipe, SlimeRecipe):
                raise TypeError(
                    f"Recipe type {recipe_type} requires SlimeRecipe, got {type(self.recipe).__name__}"
                )
            base_recipe = SlimeRecipe.get_base_recipe(self.model)
            if base_recipe is not None:
                combined = _merge_recipe(base_recipe, cast(SlimeRecipe, self.recipe))
            else:
                combined = cast(SlimeRecipe, self.recipe)
            return build_slime_app(
                training_run_id=self.training_run_id,
                slime=combined,
                model=self.model,
                dataset=self.dataset,
                checkpoint=self.checkpoint,
                name=self.training_run_id,
            )
        raise ValueError(f"Unknown recipe type: {recipe_type}")

    # ── Run-record helpers ─────────────────────────────────────────────────

    def _framework(self) -> Framework:
        if isinstance(self.recipe, SlimeRecipe):
            return Framework.SLIME
        if isinstance(self.recipe, MilesConfig):
            return Framework.MILES
        raise ValueError(f"Unknown recipe type: {type(self.recipe).__name__}")

    def _initializing_status(self) -> FrameworkStatus:
        if isinstance(self.recipe, SlimeRecipe):
            return SlimeStatus.INITIALIZING
        if isinstance(self.recipe, MilesConfig):
            return MilesStatus.INITIALIZING
        raise ValueError(f"Unknown recipe type: {type(self.recipe).__name__}")

    def _build_config_summary(self) -> dict[str, Any]:
        """Framework-specific TrainingRun.config summary."""
        model = self.model
        dataset = self.dataset
        recipe = self.recipe

        wandb = getattr(recipe, "wandb", None)
        summary: dict[str, Any] = {
            "model": {"model_name": model.model_name} if model else {},
            "wandb": (
                {"project": wandb.project, "group": wandb.group} if wandb else {}
            ),
            "dataset": {
                "hf_repo": getattr(dataset, "hf_repo", ""),
                "name": type(dataset).__name__,
            },
            "lr": getattr(recipe, "lr", None),
            "global_batch_size": getattr(recipe, "global_batch_size", None),
        }

        if isinstance(recipe, SlimeRecipe):
            from modal_training_gym.frameworks.slime.launcher import (
                _serialize_slime_params,
            )

            base_recipe = SlimeRecipe.get_base_recipe(model)
            combined = (
                _merge_recipe(base_recipe, cast(SlimeRecipe, recipe))
                if base_recipe is not None
                else cast(SlimeRecipe, recipe)
            )
            summary["recipe"] = _serialize_slime_params(
                combined, dataset=dataset, model=model
            )
        elif isinstance(recipe, MilesConfig):
            summary["recipe"] = {
                "gpu_type": recipe.gpu_type,
                "actor_num_nodes": recipe.actor_num_nodes,
                "actor_num_gpus_per_node": recipe.actor_num_gpus_per_node,
            }

        return summary

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
            framework=self._framework().value,
            model_name=getattr(self.model, "model_name", "") if self.model else "",
            dataset_name=dataset_name,
            framework_status_url=framework_status_url,
            config_path=str(config_path),
        )

    def train(self) -> TrainResult:
        """Build the app, run training, and return the TrainResult."""
        import modal

        from modal_training_gym.common.config import (
            CONFIG_PATH,
            get_framework_status_url,
        )
        from modal_training_gym.common.status_reporter import (
            enqueue_framework_status,
            flush as flush_status_reporter,
        )

        from modal_training_gym.setup import ensure_dashboard_deployed

        training_run_id = self.training_run_id

        # Auto-provision the observability dashboard the first time anyone
        # runs a training job. Idempotent and best-effort — a deploy failure
        # only costs status reporting, not the run itself.
        ensure_dashboard_deployed()

        # Resolve the dashboard URL locally so we can pass it into the
        # container — the toml lives on the user's machine, not in Modal.
        framework_status_url = get_framework_status_url() or ""

        status_display = self._build_status_display(
            training_run_id, framework_status_url, CONFIG_PATH
        )
        status_display.print_banner()

        # Write the initial TrainingRun record before the app starts so the
        # dashboard shows the run immediately (even during image build).
        created_at = int(time.time())
        run_record = TrainingRun(
            training_run_id=training_run_id,
            modal_app_id="",
            modal_app_url="",
            framework=self._framework(),
            config=self._build_config_summary(),
            framework_status=self._initializing_status(),
            created_at=created_at,
            started_at=created_at,
        )
        try:
            run_record.save()
        except RuntimeError:
            pass
        try:
            framework_status_token = _secrets.token_urlsafe(32)
            vol_put(
                MetadataStore.FRAMEWORK_STATUS_TOKENS,
                training_run_id,
                {"token": framework_status_token},
            )
        except RuntimeError:
            framework_status_token = ""
        print(f"TrainingRun recorded: {training_run_id}")

        app = self._build_app()
        result_dict = None
        modal_app_id = ""
        with modal.enable_output():
            with app.run(detach=self.detach):
                modal_app_id = app.app_id or ""
                modal_app_url = modal_app_dashboard_url(modal_app_id)
                status_display.set_modal_app_url(modal_app_url)

                # Update the record with the Modal app ID now that we have it.
                run_record.modal_app_id = modal_app_id
                run_record.modal_app_url = modal_app_url
                try:
                    run_record.save()
                except RuntimeError:
                    pass

                # Mid-flight status bumps are fire-and-forget HTTP posts to
                # the dashboard so the orchestration thread doesn't block on
                # Modal Volume writes between download.remote() /
                # convert.remote() calls. Also emits a one-line scrolling
                # status update to the terminal.
                #
                # ``is_active=False`` marks "we've queued this stage but the
                # GPU/container isn't running yet" — the Modal function
                # itself flips it to True when its body actually starts
                # (see download/convert_checkpoint in the framework
                # launchers).
                def _set_status(
                    status: FrameworkStatus, *, is_active: bool = True
                ) -> None:
                    run_record.framework_status = status
                    status_display.emit_stage(status.value)
                    enqueue_framework_status(
                        training_run_id,
                        status.value,
                        token=framework_status_token,
                        is_active=is_active,
                    )

                # Bridge mode loads HF tensors directly, so the megatron→HF
                # preconversion step (convert_checkpoint) is a no-op. For any
                # other value of megatron_to_hf_mode — including the
                # empty-string default (mbridge) — we always run the
                # pre-conversion so training starts from the torch_dist
                # layout it expects.
                megatron_to_hf_mode = getattr(self.recipe, "megatron_to_hf_mode", "")
                needs_conversion = megatron_to_hf_mode != "bridge"
                try:
                    if isinstance(self.recipe, SlimeRecipe):
                        _set_status(SlimeStatus.DOWNLOAD_MODEL, is_active=False)
                        app.download.remote(
                            training_run_id=training_run_id,
                            framework_status_url=framework_status_url,
                            framework_status_token=framework_status_token,
                        )
                        if needs_conversion:
                            _set_status(SlimeStatus.CONVERT_MODEL, is_active=False)
                            app.convert_checkpoint.remote(
                                training_run_id=training_run_id,
                                framework_status_url=framework_status_url,
                                framework_status_token=framework_status_token,
                            )
                    elif isinstance(self.recipe, MilesConfig):
                        # Miles handles model download internally inside
                        # train.remote() (_prepare_shared_inputs), so we only
                        # spawn the standalone download container when
                        # there's a non-bridge conversion to chain.
                        if needs_conversion:
                            _set_status(MilesStatus.DOWNLOAD_MODEL, is_active=False)
                            app.download.remote(
                                training_run_id=training_run_id,
                                framework_status_url=framework_status_url,
                                framework_status_token=framework_status_token,
                            )
                            _set_status(MilesStatus.CONVERT_MODEL, is_active=False)
                            app.convert_checkpoint.remote(
                                training_run_id=training_run_id,
                                framework_status_url=framework_status_url,
                                framework_status_token=framework_status_token,
                            )
                    # The remote call blocks for the whole run; poll the run
                    # record so the terminal Stage line tracks the container's
                    # phases instead of freezing at the last local stage.
                    status_display.start_polling(training_run_id)
                    result_dict = app.train.remote(
                        modal_app_id=modal_app_id,
                        modal_app_url=modal_app_url,
                        framework_status_url=framework_status_url,
                        framework_status_token=framework_status_token,
                    )
                except BaseException:
                    if result_dict is None:
                        # Re-read from the volume so we don't clobber
                        # metadata the dashboard built up during training
                        # (latest_rollout, framework_progress, etc.).
                        try:
                            run_record = TrainingRun.from_id(training_run_id)
                        except (KeyError, Exception):
                            pass
                        # A detached app survives client disconnect, so a
                        # dropped RPC (laptop sleep, network blip, Ctrl-C) is
                        # not a real failure — the container keeps training and
                        # will write its own terminal state. Don't stamp FAILED
                        # while the app is still alive on Modal.
                        remote_still_running = bool(
                            self.detach
                            and modal_app_id
                            and _app_is_running(modal_app_id)
                        )
                        # Only mark FAILED if the remote hasn't already set a
                        # terminal state (it may have completed/failed on its
                        # own while we lost the RPC connection) and the app
                        # isn't still running detached.
                        if (
                            run_record.status == TrainingRunStatus.RUNNING
                            and not remote_still_running
                        ):
                            # TODO(joy/melody): Record the exec type also in the run record.
                            run_record.status = TrainingRunStatus.FAILED
                            finished_at = int(time.time())
                            run_record.ended_at = finished_at
                            if run_record.completed_at is None:
                                run_record.completed_at = finished_at
                            run_record.duration_seconds = max(
                                0, finished_at - run_record.started_at
                            )
                            # Terminal-state write is synchronous to guarantee
                            # the failure shows up in the dashboard even if the
                            # background reporter is still draining.
                            try:
                                run_record.save()
                            except RuntimeError:
                                pass
                    raise
                finally:
                    status_display.stop_polling()
                    # Give any in-flight status POSTs a moment to land
                    # before the process exits.
                    flush_status_reporter(timeout_seconds=2.0)
        # A detached app survives client disconnect (so a closed terminal won't
        # kill a run) — but for the same reason it won't auto-stop when the run
        # finishes. Stop it ourselves once training completed successfully; on
        # interrupt/failure we leave it up so it can be inspected.
        if self.detach and modal_app_id and result_dict is not None:
            _stop_app(modal_app_id)
        if result_dict is None:
            raise RuntimeError(
                "Training app exited before returning a result. "
                "The run is detached, so it keeps running on Modal even if this "
                "client disconnects — reattach with `modal app logs <app-id>` or "
                "stop it with `modal app stop <app-id>`."
            )
        result = TrainResult(**TrainResult._parse_model_config(result_dict))
        print(f"Training complete: {result.training_run_id}")
        return result
