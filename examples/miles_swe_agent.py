"""Miles SWE-Agent training on Modal via modal-training-gym.

Mirrors the upstream ``radixark/miles/examples/swe-agent`` setup:
https://github.com/radixark/miles/blob/b1860dd264e17c96d5d92da96c957d88cfd3a1f8/examples/swe-agent/README.md

Prerequisites
-------------
* A Harbor agent server running and reachable from the Modal cluster.
  See the Miles README for how to start it.
* Modal Secrets: ``huggingface-secret`` (HF_TOKEN) and optionally
  ``wandb-secret`` (WANDB_API_KEY).
* Fill in ``AGENT_SERVER_URL`` and ``MILES_ROUTER_EXTERNAL_HOST`` below.

If you need to patch Miles or the base image, fork ``radixark/miles``, build the
image, and set ``docker_image`` to your fork's image tag.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import modal

from modal_training_gym import MilesRecipe, TrainConfig
from modal_training_gym.common.dataset import DatasetConfig
from modal_training_gym.common.errors import TrainingGymConfigError
from modal_training_gym.common.models import HFModelConfiguration


class GLM_4_7_Flash(HFModelConfiguration):
    """GLM-4.7-Flash checkpoint for the Miles agentic-coding recipe."""

    model_name = "zai-org/GLM-4.7-Flash"


class SWEAgentDataset(DatasetConfig):
    """Converts a HuggingFace JSONL into the Miles ``prompt``/``metadata`` format.

    The ``metadata`` dict carries ``instance_id`` (which Harbor maps to a task
    directory) plus the original sample fields and ``agent_name``.

    This example does not define a held-out eval split, so tell the launcher
    not to materialize a companion ``eval.*`` file.
    """

    input_key: str = "prompt"
    label_key: str = ""
    apply_chat_template: bool = False
    output_format: str = "jsonl"
    writes_eval_paths: bool = False

    hf_repo: str = "SWE-Gym/SWE-Gym"
    hf_split: str = "train"
    prompt_key: str = "problem_statement"
    agent_name: str = "mini-swe-agent"
    n_rows: int | None = None

    def _validate(self) -> None:
        # Miles only needs an ``input_key`` and a separate ``metadata`` column;
        # we skip the generic ``label_key`` requirement.
        if not self.input_key:
            raise TrainingGymConfigError(
                f"{type(self).__name__} requires `input_key` to be set."
            )

    def load(self, split: str = "all") -> list[dict[str, object]]:
        from datasets import load_dataset

        ds = load_dataset(self.hf_repo, split=self.hf_split)
        if self.n_rows:
            ds = ds.select(range(min(self.n_rows, len(ds))))

        rows: list[dict[str, object]] = []
        for i, ex in enumerate(ds):
            instance = dict(ex)
            metadata = {
                **instance,
                "agent_name": self.agent_name,
                "split": self.hf_split,
            }
            if "instance_id" not in metadata:
                metadata["instance_id"] = instance.get("id", str(i))

            prompt = instance.get(self.prompt_key, "")
            for fallback in ("problem_statement", "instruction", "prompt"):
                if not prompt:
                    prompt = instance.get(fallback, "")

            rows.append({"prompt": prompt, "metadata": metadata})
        return rows

    def prepare(self, path: str, eval_paths: dict[str, str] | None = None) -> None:
        import os

        rows = self.load()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            for row in rows:
                f.write(json.dumps(row, default=str) + "\n")

        if eval_paths:
            for eval_path in eval_paths.values():
                os.makedirs(os.path.dirname(eval_path), exist_ok=True)
                with open(eval_path, "w") as f:
                    for row in rows:
                        f.write(json.dumps(row, default=str) + "\n")


def swe_agent_reward(args, samples, **kwargs):
    """Reward is pre-computed by the Harbor agent server and stored in metadata."""
    if isinstance(samples, list):
        return [s.metadata.get("reward", 0.0) for s in samples]
    return samples.metadata.get("reward", 0.0)


def image_overlay(image: modal.Image) -> modal.Image:
    """Copy the agent helper into the container so Miles can import it by name."""
    helper = Path(__file__).resolve().parent / "miles_swe_agent_function.py"
    if helper.exists():
        image = image.add_local_file(
            str(helper),
            remote_path="/root/miles_swe_agent_function.py",
            copy=True,
        )
    return image


dataset = SWEAgentDataset(
    hf_repo="SWE-Gym/SWE-Gym",
    prompt_key="problem_statement",
    agent_name="mini-swe-agent",
    n_rows=4,  # smoke-size; remove for a real run
)

model = GLM_4_7_Flash()

recipe = MilesRecipe(
    docker_image="radixark/miles:dev-202606111336",
    miles_model_script="scripts/models/glm4.7-flash.sh",
    hf_checkpoint="zai-org/GLM-4.7-Flash",
    ref_load="/checkpoints/GLM-4.7-Flash-torch-dist",
    megatron_to_hf_mode="raw",
    actor_num_nodes=1,
    actor_num_gpus_per_node=8,
    rollout_num_gpus=8,
    tensor_model_parallel_size=4,
    pipeline_model_parallel_size=1,
    context_parallel_size=1,
    expert_model_parallel_size=8,
    expert_tensor_parallel_size=1,
    sequence_parallel=True,
    recompute_granularity="full",
    recompute_method="uniform",
    recompute_num_layers=1,
    use_dynamic_batch_size=True,
    max_tokens_per_gpu=16384,
    optimizer_cpu_offload=True,
    overlap_cpu_optimizer_d2h_h2d=True,
    use_precision_aware_optimizer=True,
    attention_backend="flash",
    num_rollout=1,  # smoke; set to 3000 for the full run
    rollout_batch_size=4,
    n_samples_per_prompt=8,
    global_batch_size=32,
    rollout_max_response_len=8192,
    rollout_temperature=0.8,
    rollout_shuffle=True,
    balance_data=True,
    lr=1e-6,
    weight_decay=0.1,
    adam_beta1=0.9,
    adam_beta2=0.98,
    lr_decay_style="constant",
    optimizer="adam",
    advantage_estimator="grpo",
    use_kl_loss=True,
    kl_loss_coef=0.01,
    kl_loss_type="low_var_kl",
    eps_clip=0.2,
    eps_clip_high=0.28,
    entropy_coef=0.0,
    save_interval=100,
    sglang_mem_fraction_static=0.7,
    sglang_tool_call_parser="glm47",
    sglang_reasoning_parser="glm45",
    rollout_num_gpus_per_engine=1,
    colocate=True,
    custom_rm_function=swe_agent_reward,
    image_overlay=image_overlay,
    environment={
        "PYTHONPATH": "/root/Megatron-LM/:/root",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "NCCL_NVLS_ENABLE": "1",
        "MILES_EXPERIMENTAL_ROLLOUT_REFACTOR": "1",
        "AGENT_SERVER_URL": os.environ.get("AGENT_SERVER_URL", ""),
        "AGENT_MODEL_NAME": os.environ.get("AGENT_MODEL_NAME", "model"),
        "MILES_ROUTER_EXTERNAL_HOST": os.environ.get("MILES_ROUTER_EXTERNAL_HOST", ""),
        "HARBOR_TASKS_DIR": os.environ.get("HARBOR_TASKS_DIR", "/root/harbor_tasks"),
    },
    extra_config={
        "custom_generate_function_path": "miles.rollout.generate_hub.agentic_tool_call.generate",
        "custom_agent_function_path": "miles_swe_agent_function.run",
        "tito_model": "glm47",
        "use_session_server": True,
        "session_server_port": [30000],
        "sglang_router_port": 31000,
        "max_seq_len": 65536,
        "metadata_key": "metadata",
        "dynamic_sampling_filter_path": "miles.rollout.filter_hub.dynamic_sampling_filters.check_no_aborted",
    },
)


def _require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise TrainingGymConfigError(
            f"{name} must be set before running this example. "
            "See the Prerequisites section in the file docstring."
        )
    return value


def main() -> None:
    recipe.environment["AGENT_SERVER_URL"] = _require_env("AGENT_SERVER_URL")
    recipe.environment["MILES_ROUTER_EXTERNAL_HOST"] = _require_env(
        "MILES_ROUTER_EXTERNAL_HOST"
    )

    result = TrainConfig(
        model=model,
        dataset=dataset,
        recipe=recipe,
    ).train()
    print(result.training_run_id)


if __name__ == "__main__":
    main()
