"""Factory that builds a Modal app for a verl RLVR/GRPO training run.

Usage (from a tutorial file):

    from modal_training_gym.common.models import Qwen3_32B
    from modal_training_gym.frameworks.verl import (
        VerlConfig, VerlFrameworkConfig, build_verl_app,
    )

    verl = VerlConfig(
        model=Qwen3_32B(),  # or subclass ModelConfiguration for a custom HF model
        dataset=...,  # DatasetConfig subclass whose prepare() writes
        wandb=...,
        framework_config=VerlFrameworkConfig(gpu="H100"),
    )

    app = build_verl_app(verl=verl)

Produces a `modal.App` with:
  - `download_model`      — snapshot_download HF weights into a volume.
  - `convert_hf_to_mcore` — convert HF → Megatron-core via verl's own script.
  - `prep_dataset`        — run `verl.dataset.prepare()`.
  - `train_multi_node`    — Ray-cluster RLVR training with `JobSubmissionClient`.
"""

from __future__ import annotations

import inspect
import subprocess

import cloudpickle
from modal import App, Image, Secret, Volume
from modal.experimental import clustered

from modal_training_gym.common import COMMON_TRAINING_GYM_TAGS
from modal_training_gym.common.framework import resolve_caller_module
from modal_training_gym.common.ray_cluster import ModalRayCluster

from .config import (
    DATA_PATH,
    MODELS_PATH,
    VerlConfig,
)

_VERL_REPO_PATH = "/root/verl"


def build_verl_app(
    *,
    verl: VerlConfig,
    name: str | None = None,
) -> App:
    app_name = name or f"verl-{type(verl).__name__.lstrip('_').lower()}"
    framework = verl.framework_config

    # Inline user-defined classes (e.g. `_Verl`) so the remote container
    # doesn't need to re-import the tutorial module.
    caller_module = resolve_caller_module()
    if caller_module is not None:
        cloudpickle.register_pickle_by_value(caller_module)

    # ── Images ───────────────────────────────────────────────────────────────
    # verl ships pre-built images; we add git + clone the repo for its
    # `scripts/converter_hf_to_mcore.py` and `examples/data_preprocess/gsm8k.py`.
    train_image = (
        Image.from_registry(framework.image)
        .apt_install(
            "git",
            "libibverbs-dev",
            "libibverbs1",
            "libhwloc15",
            "libnl-route-3-200",
        )
        .run_commands(f"git clone {framework.verl_git_url} {_VERL_REPO_PATH}")
        .uv_pip_install(_VERL_REPO_PATH)
        .entrypoint([])
        .add_local_python_source("modal_training_gym", copy=True)
    )

    download_image = (
        Image.debian_slim(python_version="3.12")
        .uv_pip_install("huggingface_hub==1.2.1")
        .env({"HF_XET_HIGH_PERFORMANCE": "1"})
        .add_local_python_source("modal_training_gym", copy=True)
    )

    # ── Volumes ──────────────────────────────────────────────────────────────
    data_volume = Volume.from_name(f"{app_name}-data", create_if_missing=True)
    checkpoints_volume = Volume.from_name(
        f"{app_name}-checkpoints", create_if_missing=True
    )
    rewards_volume = Volume.from_name(f"{app_name}-rewards", create_if_missing=True)

    data_path_str = str(DATA_PATH)
    models_path_str = str(MODELS_PATH)
    rewards_dir = "/rewards"
    uploaded_reward_path = f"{rewards_dir}/custom_reward.py"

    tags = {
        **COMMON_TRAINING_GYM_TAGS,
        "framework": "verl",
        **framework.app_tags,
    }
    app = App(app_name, tags=tags)

    # ── download_model ───────────────────────────────────────────────────────
    @app.function(
        image=download_image,
        volumes={models_path_str: checkpoints_volume},
        timeout=24 * 60 * 60,
        serialized=True,
        name="download_model",
    )
    def download_model(revision: str | None = None):
        """Snapshot_download the HF model into the checkpoints volume."""
        from huggingface_hub import snapshot_download

        assert verl.model is not None, "verl.model must be set"
        target = verl.hf_model_dir()
        snapshot_download(
            repo_id=verl.model.model_name,
            local_dir=target,
            revision=revision,
        )
        print(f"Model downloaded to {target}")
        checkpoints_volume.commit()

    # ── convert_hf_to_mcore ──────────────────────────────────────────────────
    @app.function(
        image=train_image,
        gpu=f"{framework.gpu}:2",  # verl's converter runs on a small single node.
        timeout=24 * 60 * 60,
        volumes={models_path_str: checkpoints_volume},
        serialized=True,
        name="convert_hf_to_mcore",
    )
    def convert_hf_to_mcore():
        """Convert HF weights → Megatron-core `torch_dist` via verl's script."""
        assert verl.model is not None
        checkpoints_volume.reload()
        subprocess.run(
            [
                "python",
                f"{_VERL_REPO_PATH}/scripts/converter_hf_to_mcore.py",
                "--hf_model_path",
                verl.hf_model_dir(),
                "--output_path",
                verl.mcore_model_dir(),
            ],
            check=True,
        )
        checkpoints_volume.commit()

    # ── prep_dataset ─────────────────────────────────────────────────────────
    @app.function(
        image=train_image,
        volumes={data_path_str: data_volume},
        timeout=4 * 60 * 60,
        serialized=True,
        name="prep_dataset",
    )
    def prep_dataset():
        """Run `verl.dataset.prepare()` to populate the data volume."""
        assert verl.dataset is not None, "verl.dataset must be set"
        data_volume.reload()
        verl.dataset.prepare()
        data_volume.commit()

    # ── upload_reward ────────────────────────────────────────────────────────
    @app.function(
        image=download_image,
        volumes={rewards_dir: rewards_volume},
        timeout=5 * 60,
        serialized=True,
        name="upload_reward",
    )
    def upload_reward():
        """Write `framework.reward_function_source` to the rewards volume.

        Call this once (or whenever the source changes) before `train_multi_node`.
        The resulting file is a regular `.py` importable from either a Python
        script or a Jupyter notebook, and `train_multi_node` points verl at it
        via `custom_reward_function.path=`.
        """
        import os

        if not framework.reward_function_source.strip():
            print("No reward_function_source set; nothing to upload.")
            return
        os.makedirs(rewards_dir, exist_ok=True)
        with open(uploaded_reward_path, "w") as f:
            f.write(framework.reward_function_source)
        rewards_volume.commit()
        print(
            f"Wrote reward source ({len(framework.reward_function_source)} chars) "
            f"to {uploaded_reward_path}"
        )

    # ── train_multi_node ─────────────────────────────────────────────────────
    @app.function(
        image=train_image,
        gpu=f"{framework.gpu}:{framework.gpus_per_node}",
        volumes={
            models_path_str: checkpoints_volume,
            data_path_str: data_volume,
            rewards_dir: rewards_volume,
        },
        secrets=[Secret.from_name("wandb-secret")],
        timeout=24 * 60 * 60,
        experimental_options={"efa_enabled": True},
        serialized=True,
        name="train_multi_node",
    )
    @clustered(size=framework.n_nodes, rdma=True)
    async def train_multi_node(*extra_overrides: str):
        """Multi-node GRPO training: bring up Ray, submit verl job via client.

        Any positional arg gets appended to the verl Hydra overrides, so you
        can do `modal run ...::app.train_multi_node -- trainer.total_epochs=20`.
        """
        assert verl.model is not None
        assert verl.dataset is not None

        checkpoints_volume.reload()
        data_volume.reload()

        cluster = ModalRayCluster()
        cluster.start(n_nodes=framework.n_nodes)

        if not cluster.is_head:
            await cluster.wait_forever()
            return

        # Resolve the reward file path. Priority order: uploaded source on
        # the rewards volume → explicit `reward_function_path` override →
        # shipped `gsm8k_reward` default.
        import os as _os

        if (
            framework.reward_function_source.strip()
            and _os.path.exists(uploaded_reward_path)
        ):
            reward_path = uploaded_reward_path
        elif framework.reward_function_path:
            reward_path = framework.reward_function_path
        else:
            import importlib.util

            spec = importlib.util.find_spec(
                "modal_training_gym.frameworks.verl.gsm8k_reward"
            )
            assert spec is not None and spec.origin is not None
            reward_path = spec.origin
        print(f"Using reward function at {reward_path}")

        overrides = verl.cli_args(reward_function_path=reward_path)
        overrides.extend(extra_overrides)

        verl_cmd = [
            "python",
            "-u",
            "-m",
            "verl.trainer.main_ppo",
            "--config-path=config",
            "--config-name=ppo_megatron_trainer.yaml",
            *overrides,
        ]

        async with cluster.forward_dashboard() as tunnel:
            print(f"Ray dashboard: {tunnel.url}")
            entrypoint = " ".join(verl_cmd)
            print(f"Submitting verl job: {entrypoint}")
            await cluster.submit_and_tail(entrypoint)

        checkpoints_volume.commit()

    # Expose registered functions as attributes (notebook ergonomics).
    for tag, fn in app.registered_functions.items():
        setattr(app, tag, fn)

    return app
