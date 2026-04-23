"""Factory that builds a Modal app for Harbor + Miles RL training.

Usage (from a tutorial file):

    from modal_training_gym.frameworks.harbor import (
        HarborConfig, HarborFrameworkConfig, build_harbor_app,
    )

    harbor = HarborConfig(
        model=Qwen3_4B(),
        dataset=MyHarborDataset(),
        framework_config=HarborFrameworkConfig(
            gpu="H100",
            n_nodes=2,
            agent_import_path="my_agent:MyHarborAgent",
            recipe_args=\"\"\"
                --num-layers 36
                --hidden-size 2560
                ...
            \"\"\",
        ),
    )

    app = harbor.build_app()

Exposes ``app.download_model``, ``app.prepare_dataset``,
``app.train_multi_node``, and ``app.eval_harbor``.
"""

from __future__ import annotations

import asyncio
import json
import os
import pathlib
import shlex
import time
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Any

import cloudpickle
from modal import App, Image, Secret, Volume
from modal.experimental import clustered

from modal_training_gym.common import COMMON_TRAINING_GYM_TAGS
from modal_training_gym.common.framework import (
    TOOLS_LOCAL_PATH,
    TOOLS_REMOTE_PATH,
    resolve_caller_module,
)
from modal_training_gym.common.ray_cluster import ModalRayCluster

from modal_training_gym.frameworks.miles.config import model_training_cli_args

from .config import (
    CHECKPOINTS_PATH,
    DATA_PATH,
    HARBOR_TRAIN_JSONL,
    HF_CACHE_PATH,
    HarborConfig,
    MILES_SRC_PATH,
    ROLLOUT_PROXY_PORT,
    ROLLOUT_ROUTER_PORT,
)
from .observability import (
    OBSERVABILITY_COMMIT_INTERVAL_ENV,
    OBSERVABILITY_ENABLED_ENV,
    OBSERVABILITY_MOUNT,
    OBSERVABILITY_ROOT_ENV,
    OBSERVABILITY_RUN_ID_ENV,
    OBSERVABILITY_VOLUME,
    OBSERVABILITY_VOLUME_NAME_ENV,
    initialize_run_manifest,
    update_run_manifest,
)

_REMOTE_TRAIN_SCRIPT = f"{MILES_SRC_PATH}/train.py"

_HARBOR_GENERATE_FN = "miles.rollout.generate_hub.agentic_tool_call.generate"
_HARBOR_AGENT_FUNCTION = "modal_training_gym.frameworks.harbor.agent_function.run"
_HARBOR_REWARD_FUNC = "modal_training_gym.frameworks.harbor.reward.reward_func"
_HARBOR_ROLLOUT_FN = "modal_training_gym.frameworks.harbor.reward.RolloutFn"


def _build_training_jsonl(
    task_root: str,
    output_path: Path,
    *,
    task_glob: str = "*",
    instruction_path: str = "instruction.md",
    agent_kwargs: dict[str, Any] | None = None,
) -> int:
    """Scan Harbor task directories and write a Miles-compatible JSONL."""
    task_root_path = Path(task_root)
    if not task_root_path.exists():
        raise FileNotFoundError(f"Task root does not exist: {task_root}")

    task_dirs = sorted(p for p in task_root_path.glob(task_glob) if p.is_dir())
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with output_path.open("w") as fh:
        for task_dir in task_dirs:
            prompt_file = task_dir / instruction_path
            if not prompt_file.exists():
                continue
            prompt = prompt_file.read_text().strip()
            if not prompt:
                continue
            metadata: dict[str, Any] = {
                "harbor_task_path": task_dir.as_posix(),
                "harbor_task_name": task_dir.name,
            }
            if agent_kwargs:
                metadata["harbor_agent_kwargs"] = agent_kwargs
            fh.write(json.dumps({"prompt": prompt, "metadata": metadata}) + "\n")
            count += 1

    if count == 0:
        raise ValueError(
            f"No Harbor tasks found in {task_root} with glob={task_glob!r} "
            f"and instruction_path={instruction_path!r}"
        )
    print(f"Built training JSONL with {count} tasks at {output_path}")
    return count


async def _proxy_stream(
    reader: asyncio.StreamReader, writer: asyncio.StreamWriter
) -> None:
    try:
        while not reader.at_eof():
            chunk = await reader.read(64 * 1024)
            if not chunk:
                break
            writer.write(chunk)
            await writer.drain()
    finally:
        writer.close()
        with suppress(Exception):
            await writer.wait_closed()


@asynccontextmanager
async def _rollout_proxy(*, target_host: str, target_port: int, listen_port: int):
    """TCP proxy forwarding ``listen_port`` → ``target_host:target_port``.

    Sandboxes reach the model server through a public Modal tunnel that
    terminates at ``listen_port``; the proxy forwards traffic to the
    internal Ray cluster rollout server.
    """

    async def _handle(
        client_reader: asyncio.StreamReader,
        client_writer: asyncio.StreamWriter,
    ) -> None:
        upstream_writer = None
        try:
            upstream_reader, upstream_writer = await asyncio.open_connection(
                target_host, target_port
            )
            up_task = asyncio.create_task(_proxy_stream(upstream_reader, client_writer))
            down_task = asyncio.create_task(
                _proxy_stream(client_reader, upstream_writer)
            )
            done, pending = await asyncio.wait(
                {up_task, down_task}, return_when=asyncio.FIRST_COMPLETED
            )
            for task in pending:
                task.cancel()
            for task in done:
                exc = task.exception()
                if exc and not isinstance(exc, asyncio.CancelledError):
                    raise exc
        except (ConnectionRefusedError, ConnectionResetError, OSError):
            pass
        finally:
            if upstream_writer is not None:
                upstream_writer.close()
                with suppress(Exception):
                    await upstream_writer.wait_closed()
            client_writer.close()
            with suppress(Exception):
                await client_writer.wait_closed()

    server = await asyncio.start_server(_handle, host="0.0.0.0", port=listen_port)
    try:
        yield
    finally:
        server.close()
        await server.wait_closed()


def build_harbor_app(
    *,
    harbor: HarborConfig,
    name: str | None = None,
) -> App:
    """Return a Modal App with Harbor + Miles training functions."""
    app_name = name or f"harbor-{type(harbor).__name__.lstrip('_').lower()}"
    framework = harbor.framework_config

    caller_module = resolve_caller_module()
    if caller_module is not None:
        cloudpickle.register_pickle_by_value(caller_module)

    caller_script = None
    if caller_module is not None:
        mod_file = getattr(caller_module, "__file__", None)
        if mod_file:
            caller_script = os.path.abspath(mod_file)

    # ── Image ────────────────────────────────────────────────────────────────
    image = Image.from_registry(framework.miles_image).entrypoint([])
    for cmd in framework.image_run_commands:
        image = image.run_commands(cmd)
    if framework.miles_src_commit:
        image = image.run_commands(
            "uv pip uninstall --system miles 2>/dev/null || true",
            f"git clone https://github.com/radixark/miles.git {MILES_SRC_PATH}",
            f"cd {MILES_SRC_PATH} && git checkout {framework.miles_src_commit}",
            f"cd {MILES_SRC_PATH} && uv pip install --system -e .",
        )
    if framework.harbor_install_command:
        image = image.run_commands(framework.harbor_install_command)
    image = image.add_local_python_source(
        "modal_training_gym", copy=True
    ).add_local_dir(TOOLS_LOCAL_PATH, remote_path=TOOLS_REMOTE_PATH, copy=True)
    if caller_script is not None:
        caller_remote_path = (
            f"/root/{os.path.splitext(os.path.basename(caller_script))[0]}.py"
        )
        image = image.add_local_file(
            caller_script, remote_path=caller_remote_path, copy=True
        )

    # ── Volumes ──────────────────────────────────────────────────────────────
    hf_cache_volume = Volume.from_name("huggingface-cache", create_if_missing=True)
    data_volume = Volume.from_name(f"{app_name}-data", create_if_missing=True)
    checkpoints_volume = Volume.from_name(
        f"{app_name}-checkpoints", create_if_missing=True
    )
    obs_volume = Volume.from_name(OBSERVABILITY_VOLUME, create_if_missing=True)

    hf_cache_str = str(HF_CACHE_PATH)
    data_str = str(DATA_PATH)
    checkpoints_str = str(CHECKPOINTS_PATH)
    obs_str = OBSERVABILITY_MOUNT

    # ── Nested Modal env (for sandbox creation) ──────────────────────────────
    nested_modal_env = {
        "MODAL_ENVIRONMENT": os.environ.get("MODAL_ENVIRONMENT", ""),
        "MODAL_TOKEN_ID": os.environ.get("MODAL_TOKEN_ID", ""),
        "MODAL_TOKEN_SECRET": os.environ.get("MODAL_TOKEN_SECRET", ""),
        "HARBOR_USE_LOCAL_TASKS": "1",
    }

    tags = {
        **COMMON_TRAINING_GYM_TAGS,
        "_modal_framework": "harbor",
        **framework.app_tags,
    }
    app = App(app_name, tags=tags)
    gpu_spec = f"{framework.gpu}:{framework.gpus_per_node}"

    # ── download_model ───────────────────────────────────────────────────────
    @app.function(
        image=image,
        volumes={
            hf_cache_str: hf_cache_volume,
            checkpoints_str: checkpoints_volume,
        },
        secrets=[Secret.from_name("huggingface-secret")],
        timeout=24 * 60 * 60,
        serialized=True,
        name="download_model",
    )
    def download_model():
        """Download model weights via the attached ModelConfiguration's hook."""
        assert harbor.model is not None, "harbor.model must be set"
        hf_cache_volume.reload()
        checkpoints_volume.reload()
        harbor.model.download_model()
        hf_cache_volume.commit()
        checkpoints_volume.commit()

    # ── prepare_dataset ──────────────────────────────────────────────────────
    @app.function(
        image=image,
        volumes={data_str: data_volume},
        timeout=4 * 60 * 60,
        serialized=True,
        name="prepare_dataset",
    )
    def prepare_dataset():
        assert harbor.dataset is not None, "harbor.dataset must be set"
        data_volume.reload()
        harbor.dataset.prepare()
        _build_training_jsonl(
            framework.task_root,
            HARBOR_TRAIN_JSONL,
            task_glob=framework.task_glob,
            instruction_path=framework.instruction_path,
            agent_kwargs=framework.agent_kwargs or None,
        )
        data_volume.commit()

    # ── train_multi_node ─────────────────────────────────────────────────────
    @app.function(
        image=image,
        gpu=gpu_spec,
        env=nested_modal_env,
        volumes={
            hf_cache_str: hf_cache_volume,
            data_str: data_volume,
            checkpoints_str: checkpoints_volume,
            obs_str: obs_volume,
        },
        secrets=[
            Secret.from_name("huggingface-secret"),
            *([] if harbor.wandb is None else [Secret.from_name("wandb-secret")]),
        ],
        timeout=24 * 60 * 60,
        experimental_options={"efa_enabled": True},
        serialized=True,
        name="train_multi_node",
    )
    @clustered(size=framework.n_nodes, rdma=True)
    async def train_multi_node(*extra_argv: str):
        """Multi-node Harbor + Miles training.

        Brings up a Ray cluster, starts a rollout proxy with a public
        tunnel (so Harbor sandboxes can reach the model), and submits
        the Miles training job.
        """
        import modal

        assert harbor.model is not None, "harbor.model must be set"
        has_arch = (
            harbor.model is not None
            and harbor.model.architecture is not None
            and harbor.model.architecture.num_layers > 0
        )
        if not has_arch and not framework.recipe_args.strip() and not extra_argv:
            raise RuntimeError(
                "No model architecture found and recipe_args is empty. Either "
                "use a model with a ModelArchitecture (e.g. Qwen3_4B) or set "
                "recipe_args to the Miles CLI args for your model."
            )
        if not framework.agent_import_path:
            raise RuntimeError(
                "HarborFrameworkConfig.agent_import_path is empty — set it to the "
                "import path of your Harbor agent class (e.g. 'my_agent:MyAgent')."
            )

        await checkpoints_volume.reload.aio()
        await data_volume.reload.aio()

        cluster = ModalRayCluster()
        cluster.start(n_nodes=framework.n_nodes)

        if not cluster.is_head:
            await cluster.wait_forever()
            return

        assert harbor.model is not None
        if harbor.model.model_path:
            model_path = str(harbor.model.model_path)
        else:
            from huggingface_hub import snapshot_download

            try:
                model_path = snapshot_download(
                    harbor.model.model_name, local_files_only=True
                )
            except FileNotFoundError as exc:
                raise RuntimeError(
                    f"Model {harbor.model.model_name} not present in HF cache. "
                    f"Run `app.download_model` first."
                ) from exc

        run_id = f"{app_name}-{int(time.time())}"
        checkpoint_dir = f"{checkpoints_str}/{run_id}"

        obs_base = Path(obs_str)
        await obs_volume.reload.aio()
        initialize_run_manifest(obs_base, {
            "run_id": run_id,
            "app_name": app_name,
            "model_name": harbor.model.model_name,
            "agent_import_path": framework.agent_import_path,
            "status": "running",
            "started_at": __import__("datetime").datetime.now(
                __import__("datetime").timezone.utc
            ).isoformat(),
        })
        await obs_volume.commit.aio()

        model_arch_args: list[str] = []
        if harbor.model and harbor.model.architecture:
            model_arch_args = harbor.model.architecture.to_megatron_args()

        argv: list[str] = [
            "python3",
            _REMOTE_TRAIN_SCRIPT,
            *framework.cli_args(),
            *model_arch_args,
            *(model_training_cli_args(harbor.model) if harbor.model else []),
            *framework.parsed_recipe_args(),
            *extra_argv,
            "--train-backend",
            "megatron",
            "--hf-checkpoint",
            model_path,
            "--ref-load",
            model_path,
            "--save",
            checkpoint_dir,
            "--actor-num-nodes",
            str(framework.resolved_actor_nodes()),
            "--actor-num-gpus-per-node",
            str(framework.gpus_per_node),
            "--num-gpus-per-node",
            str(framework.gpus_per_node),
            # Harbor integration: the agentic generate function registers
            # --custom-agent-function-path via its add_arguments hook
            # and requires --use-session-server for TITO session tracing.
            "--prompt-data",
            str(HARBOR_TRAIN_JSONL),
            "--custom-generate-function-path",
            _HARBOR_GENERATE_FN,
            "--custom-agent-function-path",
            _HARBOR_AGENT_FUNCTION,
            "--custom-rm-path",
            _HARBOR_REWARD_FUNC,
            "--rollout-function-path",
            _HARBOR_ROLLOUT_FN,
            "--use-session-server",
            "--generate-multi-samples",
        ]

        if framework.colocate:
            argv.append("--colocate")
        else:
            rollout_gpus = framework.resolved_rollout_num_gpus()
            assert rollout_gpus is not None
            argv.extend(["--rollout-num-gpus", str(rollout_gpus)])

        if framework.custom_config_yaml.strip():
            cfg_path = f"/tmp/{run_id}-overrides.yaml"
            pathlib.Path(cfg_path).write_text(framework.custom_config_yaml)
            argv.extend(["--custom-config-path", cfg_path])

        wandb_key = os.environ.get("WANDB_API_KEY", "")
        if harbor.wandb is not None and wandb_key:
            argv.extend(["--use-wandb", "--wandb-key", wandb_key])

        SESSION_SERVER_PORT = 5578
        SESSION_PROXY_PORT = 30002

        runtime_env: dict[str, Any] = {
            "env_vars": {
                "MASTER_ADDR": cluster.head_addr,
                "MILES_HOST_IP": cluster.head_addr,
                "no_proxy": cluster.head_addr,
                "PYTHONPATH": f"{MILES_SRC_PATH}:/root/Megatron-LM",
                "CUDA_DEVICE_MAX_CONNECTIONS": "1",
                "NCCL_ALGO": "Ring",
                "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                "HARBOR_AGENT_IMPORT_PATH": framework.agent_import_path,
                "HARBOR_AGENT_KWARGS": json.dumps(framework.agent_kwargs),
                "AGENT_MODEL_NAME": framework.agent_model_name,
                "HARBOR_USE_LOCAL_TASKS": "1",
                "HARBOR_SANDBOX_TIMEOUT_SECS": str(
                    framework.sandbox_timeout_secs
                ),
                "HARBOR_SANDBOX_IDLE_TIMEOUT_SECS": str(
                    framework.sandbox_idle_timeout_secs
                ),
                "MODAL_ENVIRONMENT": os.environ.get("MODAL_ENVIRONMENT", ""),
                "MODAL_TOKEN_ID": os.environ.get("MODAL_TOKEN_ID", ""),
                "MODAL_TOKEN_SECRET": os.environ.get("MODAL_TOKEN_SECRET", ""),
                OBSERVABILITY_ENABLED_ENV: "1",
                OBSERVABILITY_RUN_ID_ENV: run_id,
                OBSERVABILITY_ROOT_ENV: obs_str,
                OBSERVABILITY_VOLUME_NAME_ENV: OBSERVABILITY_VOLUME,
                OBSERVABILITY_COMMIT_INTERVAL_ENV: "15",
            }
        }
        if framework.environment_import_path:
            runtime_env["env_vars"]["HARBOR_ENVIRONMENT_IMPORT_PATH"] = (
                framework.environment_import_path
            )
        if wandb_key:
            runtime_env["env_vars"]["WANDB_API_KEY"] = wandb_key

        # The agentic generate function uses a session server for TITO
        # tracing. Sandboxes reach the session server via a public
        # tunnel through a local TCP proxy.
        async with _rollout_proxy(
            target_host=cluster.head_addr,
            target_port=SESSION_SERVER_PORT,
            listen_port=SESSION_PROXY_PORT,
        ):
            async with modal.forward(SESSION_PROXY_PORT) as session_tunnel:
                public_base_url = session_tunnel.url.rstrip("/")
                runtime_env["env_vars"]["HARBOR_PUBLIC_BASE_URL"] = (
                    public_base_url
                )

                entrypoint = shlex.join(argv)
                async with cluster.forward_dashboard() as tunnel:
                    print(f"Ray dashboard: {tunnel.url}")
                    print(f"Public session server URL: {public_base_url}")
                    print(f"Miles entrypoint: {entrypoint}")
                    status = await cluster.submit_and_tail(
                        entrypoint, runtime_env=runtime_env
                    )
                    print(f"Final status: {status}")

        await checkpoints_volume.commit.aio()

        from datetime import datetime, timezone

        final_status = "completed" if status == "SUCCEEDED" else "failed"
        update_run_manifest(
            obs_base,
            run_id,
            status=final_status,
            ended_at=datetime.now(timezone.utc).isoformat(),
        )
        await obs_volume.commit.aio()

        return {"run_id": run_id, "checkpoint_dir": checkpoint_dir}

    # ── eval_harbor ──────────────────────────────────────────────────────────
    @app.function(
        image=image,
        env=nested_modal_env,
        volumes={data_str: data_volume},
        timeout=24 * 60 * 60,
        serialized=True,
        name="eval_harbor",
    )
    def eval_harbor(server_url: str, max_concurrency: int = 4):
        """Run Harbor evaluation against a served model checkpoint.

        Reads the training JSONL to discover tasks, then runs Harbor
        trials concurrently against the provided server URL.
        """
        from .eval import run_harbor_eval, write_eval_artifacts

        data_volume.reload()
        jsonl_path = HARBOR_TRAIN_JSONL
        if not jsonl_path.exists():
            raise FileNotFoundError(
                f"Training JSONL not found at {jsonl_path}. Run prepare_dataset first."
            )

        records = [
            json.loads(line)
            for line in jsonl_path.read_text().splitlines()
            if line.strip()
        ]
        print(f"Loaded {len(records)} task records from {jsonl_path}")

        output_dir = DATA_PATH / "evals" / f"{app_name}-{int(time.time())}"
        summary, results = asyncio.run(
            run_harbor_eval(
                records=records,
                server_url=server_url,
                agent_import_path=framework.agent_import_path,
                agent_model_name=framework.agent_model_name,
                agent_kwargs=framework.agent_kwargs,
                sandbox_timeout_secs=framework.sandbox_timeout_secs,
                sandbox_idle_timeout_secs=framework.sandbox_idle_timeout_secs,
                environment_import_path=framework.environment_import_path,
                max_concurrency=max_concurrency,
            )
        )
        write_eval_artifacts(output_dir, summary, results)
        data_volume.commit()

        print(
            f"[eval-summary] tasks={summary['task_count']} ok={summary['ok_count']} "
            f"failed={summary['failure_count']} reward_mean={summary['reward_mean']:.3f}"
        )
        print(f"[eval-summary] artifacts={output_dir}")
        return summary

    for tag, fn in app.registered_functions.items():
        setattr(app, tag, fn)

    return app
