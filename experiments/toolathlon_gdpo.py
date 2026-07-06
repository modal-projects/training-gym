"""Toolathlon + GDPO: Qwen 3.6 35B with dual-reward Group Dual-clip Policy Optimization.

GDPO (Group Dual-clip Policy Optimization, arxiv:2601.05242) decouples the task
reward and a length penalty into **two independent reward dimensions**. Each
dimension gets its own GRPO-style group-normalized advantage; the policy update
uses the sum of per-dimension advantages with dual-clip PPO (``eps_clip_c``).

Compared to the vanilla Toolathlon recipe (``toolathlon.py``):
  - The generate function no longer blends the length penalty into a single
    scalar reward. ``sample.reward = task_reward`` (raw 0/1 pass/fail).
  - A GDPO custom advantage function
    (``modal_training_gym.frameworks.slime.gdpo.gdpo_compute_advantages``)
    computes per-dimension advantages and sums them.
  - The recipe enables ``eps_clip_c = 3.0`` for GDPO dual-clip.

Run::

    cd <repo-root>
    modal run experiments/toolathlon_gdpo.py --experiment-name my-gdpo-run
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import field
from pathlib import Path
from typing import Any

import modal
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from modal_training_gym.common.training_group import TrainingGroup
from modal_training_gym.train_recipes.slime_recipe import SlimeRecipe

app = modal.App("toolathlon-gdpo")

try:
    from modal_training_gym import HarborDataset, Qwen3_6_35B, TrainConfig, WandbConfig
except ImportError:
    HarborDataset = object  # type: ignore[assignment,misc]

# Import shared infrastructure from the base toolathlon module.
from toolathlon import (  # type: ignore[import-untyped]  # noqa: E402
    EPISODE_TIMEOUT_SEC,
    LENGTH_PENALTY_FREE_TOKENS,
    LENGTH_PENALTY_MAX_COST,
    LENGTH_PENALTY_MAX_TOKENS,
    MAX_TOKENS_PER_GPU,
    MINI_SWE_STEP_LIMIT,
    MINI_SWE_THINKING_BUDGET,
    QWEN3_TOOL_STOP_TOKEN_IDS,
    ROLLOUT_CONTEXT_LEN,
    ROLLOUT_RESPONSE_LEN,
    TEMPERATURE,
    ToolathlonDataset,
    ToolathlonExitStatus,
    _exception_summary,
    _image_overlay as _base_image_overlay,
    _length_penalized_reward,
    _sample_metadata,
    _training_segments,
    make_toolathlon_eval_fn,
    run_mini_swe_toolathlon_episode,
    toolathlon_eval_fn,
)


# ─────────────────────────────────────────────────────────────────────────────
# GDPO-specific generate function
# ─────────────────────────────────────────────────────────────────────────────

# Cache the tokenizer (mirrors the base toolathlon helper).
_TOKENIZER_CACHE: dict[str, Any] = {}


def _tokenizer(hf_checkpoint: str):
    if hf_checkpoint not in _TOKENIZER_CACHE:
        from transformers import AutoTokenizer

        _TOKENIZER_CACHE[hf_checkpoint] = AutoTokenizer.from_pretrained(
            hf_checkpoint, trust_remote_code=True
        )
    return _TOKENIZER_CACHE[hf_checkpoint]


async def toolathlon_gdpo_generate(
    args,
    sample,
    sampling_params,
    evaluation: bool = False,
    step_limit: int = MINI_SWE_STEP_LIMIT,
    max_new_tokens: int = ROLLOUT_RESPONSE_LEN,
    mini_swe_command_timeout_sec: int = 120,
):
    """Run a live Toolathlon episode and pack it into a slime Sample.

    GDPO variant: ``sample.reward = task_reward`` (raw 0/1); the length penalty
    is handled separately by the GDPO advantage function
    (``gdpo_compute_advantages``), which computes per-dimension advantages.
    """
    from slime.utils.types import Sample  # type: ignore[import-not-found]

    del evaluation

    metadata = _sample_metadata(sample)

    try:
        payload = await asyncio.wait_for(
            run_mini_swe_toolathlon_episode(
                base_url=f"http://{args.sglang_router_ip}:{args.sglang_router_port}",
                model_name="model",
                metadata=metadata,
                sampling_params=sampling_params,
                step_limit=step_limit,
                max_new_tokens=max_new_tokens,
                mini_swe_command_timeout_sec=mini_swe_command_timeout_sec,
            ),
            timeout=EPISODE_TIMEOUT_SEC,
        )
    except asyncio.TimeoutError:
        print(
            "[toolathlon-gdpo] episode timeout "
            f"task={metadata.get('task_name')!r} timeout={EPISODE_TIMEOUT_SEC}s",
            flush=True,
        )
        payload = {
            "reward": 0.0,
            "exit_status": ToolathlonExitStatus.TIMEOUT,
            "trajectory_messages": [],
        }
    except Exception as exc:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        print(
            "[toolathlon-gdpo] episode exception "
            f"task={metadata.get('task_name')!r} error={_exception_summary(exc)}",
            flush=True,
        )
        payload = {
            "reward": 0.0,
            "exit_status": f"exception: {_exception_summary(exc)}",
            "trajectory_messages": [],
        }

    tokenizer = _tokenizer(args.hf_checkpoint)
    prompt_text = (
        sample.prompt
        if isinstance(sample.prompt, str)
        else json.dumps(sample.prompt, sort_keys=True)
    )
    prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)

    response_tokens: list[int] = []
    loss_mask: list[int] = []
    trajectory_messages = payload.get("trajectory_messages", [])
    if not isinstance(trajectory_messages, list):
        trajectory_messages = []
    segments, assistant_turns = _training_segments(trajectory_messages)
    for text, trainable in segments:
        ids = tokenizer.encode(text, add_special_tokens=False)
        response_tokens.extend(ids)
        loss_mask.extend([trainable] * len(ids))

    if not response_tokens:
        eos = getattr(tokenizer, "eos_token_id", None) or 0
        response_tokens, loss_mask = [eos], [0]

    # Megatron context-parallel packing needs (prompt + response) divisible by 2*CP.
    cp = getattr(args, "context_parallel_size", 1) or 1
    max_seq = getattr(args, "max_tokens_per_gpu", 8192) * cp
    raw_response_length = len(response_tokens)
    response_budget = max(1, max_seq - len(prompt_tokens))
    response_tokens = response_tokens[:response_budget]
    loss_mask = loss_mask[: len(response_tokens)]
    align = 2 * cp
    remainder = (len(prompt_tokens) + len(response_tokens)) % align
    if remainder:
        eos = getattr(tokenizer, "eos_token_id", None) or 0
        if len(prompt_tokens) + len(response_tokens) + (align - remainder) <= max_seq:
            response_tokens += [eos] * (align - remainder)
            loss_mask += [0] * (align - remainder)
        else:
            response_tokens = response_tokens[: len(response_tokens) - remainder]
            loss_mask = loss_mask[: len(loss_mask) - remainder]

    sample.response = "\n\n".join(assistant_turns)
    sample.prompt_length = len(prompt_tokens)
    sample.response_length = len(response_tokens)
    sample.tokens = prompt_tokens + response_tokens
    sample.loss_mask = loss_mask

    # ── GDPO: raw task reward only ────────────────────────────────────────
    # Unlike the vanilla Toolathlon recipe which blends length penalty into a
    # single scalar, GDPO keeps rewards separate. sample.reward carries the
    # raw task reward; the GDPO advantage function computes the length penalty
    # dimension from loss_mask at advantage-computation time.
    assistant_tokens = sum(loss_mask)
    task_reward = float(payload.get("reward", 0.0))
    sample.reward = task_reward

    sample.status = Sample.Status.COMPLETED
    sample.metadata = {
        **metadata,
        **payload,
        "reward": task_reward,
        "task_reward": task_reward,
        # For logging: what the blended reward WOULD have been under vanilla.
        "vanilla_penalized_reward": _length_penalized_reward(
            task_reward, assistant_tokens
        ),
        "training_assistant_tokens": assistant_tokens,
        "exit_status": payload.get("exit_status"),
        "training_response_source": "assistant_action_turns",
        "training_assistant_turns": len(assistant_turns),
        "trajectory_message_count": len(trajectory_messages),
        "training_token_limit": max_seq,
        "training_raw_response_length": raw_response_length,
        "training_tokens_truncated": raw_response_length > len(response_tokens),
    }
    return sample


async def gdpo_reward_func(args, samples, **kwargs):
    """Pass through ``sample.reward`` (raw task reward); slime feeds it
    into the GDPO advantage function for per-dimension advantage computation.
    """
    if isinstance(samples, (list, tuple)):
        return [float(getattr(s, "reward", 0.0) or 0.0) for s in samples]
    return float(getattr(samples, "reward", 0.0) or 0.0)


def gdpo_rollout_log(
    rollout_id, args, samples, rollout_extra_metrics, rollout_time
) -> bool:
    """Log GDPO-specific metrics: task pass rate, would-be length penalty, and
    per-dimension statistics."""
    del args, rollout_time

    latencies = []
    task_rewards = []
    vanilla_penalized_rewards = []
    response_lengths = []
    assistant_tokens_list = []
    exit_status_counts: dict[str, int] = {}
    truncated_count = 0
    for sample in samples:
        task_rewards.append(float(getattr(sample, "reward", 0.0) or 0.0))
        rl = getattr(sample, "response_length", None)
        if isinstance(rl, (int, float)):
            response_lengths.append(float(rl))
        metadata = getattr(sample, "metadata", None)
        if not isinstance(metadata, dict):
            continue
        vpr = metadata.get("vanilla_penalized_reward")
        if vpr is not None:
            vanilla_penalized_rewards.append(float(vpr))
        at = metadata.get("training_assistant_tokens")
        if isinstance(at, (int, float)):
            assistant_tokens_list.append(float(at))
        exit_status = str(metadata.get("exit_status") or "unknown")
        exit_status_counts[exit_status] = exit_status_counts.get(exit_status, 0) + 1
        if metadata.get("training_tokens_truncated"):
            truncated_count += 1
        try:
            latencies.append(float(metadata["tool_execution_latency"]))
        except (KeyError, TypeError, ValueError):
            continue

    if not latencies and not task_rewards:
        return False

    metrics: dict[str, float] = {}
    if task_rewards:
        metrics["toolathlon_pass_rate"] = sum(task_rewards) / len(task_rewards)
        metrics["toolathlon_truncation_rate"] = truncated_count / len(task_rewards)
    if vanilla_penalized_rewards:
        metrics["toolathlon_vanilla_penalized_reward_mean"] = sum(
            vanilla_penalized_rewards
        ) / len(vanilla_penalized_rewards)
    if response_lengths:
        metrics["toolathlon_response_length_mean"] = sum(response_lengths) / len(
            response_lengths
        )
    if assistant_tokens_list:
        metrics["toolathlon_assistant_tokens_mean"] = sum(assistant_tokens_list) / len(
            assistant_tokens_list
        )
        # Compute what the GDPO length penalty dimension looks like.
        length_penalties = [
            _length_penalized_reward(1.0, int(at)) - 1.0 for at in assistant_tokens_list
        ]
        metrics["toolathlon_gdpo_length_penalty_mean"] = sum(length_penalties) / len(
            length_penalties
        )
    if latencies:
        metrics["tool_execution_latency"] = sum(latencies) / len(latencies)
    for status, count in exit_status_counts.items():
        metrics[f"toolathlon_exit_status/{status}"] = count

    if isinstance(rollout_extra_metrics, dict):
        rollout_extra_metrics.update(metrics)

    try:
        wandb = __import__("wandb")
    except ImportError:
        return False

    if getattr(wandb, "run", None) is not None:
        wandb.log(metrics, step=rollout_id, commit=False)
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Image overlay: base Toolathlon deps + ship base module for remote imports
# ─────────────────────────────────────────────────────────────────────────────


def _gdpo_image_overlay(image):
    """Layer the base Toolathlon deps and add toolathlon.py so rollout workers
    can import shared infrastructure (mini-swe agent, episode runner, etc.)."""
    image = _base_image_overlay(image)
    # Ship the base toolathlon module so the GDPO generate function can import
    # run_mini_swe_toolathlon_episode and helpers on the remote side.
    base_toolathlon = Path(__file__).resolve().parent.parent / "toolathlon.py"
    if base_toolathlon.is_file():
        image = image.add_local_file(
            str(base_toolathlon),
            remote_path="/root/toolathlon.py",
            copy=True,
        )
    return image


# ─────────────────────────────────────────────────────────────────────────────
# GDPO Recipe
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class ToolathlonGDPOQwen3_6_35bRecipe(SlimeRecipe):
    """Qwen3.6-35B-A3B + Toolathlon + GDPO (dual-reward, dual-clip).

    Differences from ``ToolathlonQwen3_6_35bRecipe``:
      - ``eps_clip_c = 3.0``: GDPO dual-clip for negative advantages.
      - ``custom_advantage_function_path``: per-dimension GRPO advantages
        (task reward + length penalty), summed for the policy update.
      - ``toolathlon_gdpo_generate``: raw task reward only (no blending).
    """

    # ── Infra (same as base Toolathlon recipe) ────────────────────────────
    gpu_type: str = "H200"
    slime_model_script: str = "scripts/models/qwen3.5-35B-A3B.sh"
    hf_checkpoint: str = "Qwen/Qwen3.6-35B-A3B"
    async_mode: bool = True
    train_function_kwargs: dict[str, int] = field(
        default_factory=lambda: {"ephemeral_disk": 1_048_576}
    )

    colocate: bool = False
    actor_num_nodes: int = 1
    actor_num_gpus_per_node: int = 8
    rollout_num_gpus: int = 32

    # ── Rollout ───────────────────────────────────────────────────────────
    num_rollout: int = 40
    rollout_batch_size: int = 8
    rollout_num_gpus_per_engine: int = 8
    rollout_max_response_len: int = ROLLOUT_RESPONSE_LEN
    rollout_max_context_len: int = ROLLOUT_CONTEXT_LEN
    rollout_temperature: float = 1.0
    rollout_stop_token_ids: list[int] | None = field(
        default_factory=lambda: list(QWEN3_TOOL_STOP_TOKEN_IDS)
    )
    global_batch_size: int = 64
    sglang_mem_fraction_static: float = 0.75
    sglang_ep_size: int | None = 4
    sglang_cuda_graph_bs: list[int] | None = field(
        default_factory=lambda: [1, 2, 4, 8] + list(range(16, 257, 8))
    )

    sglang_speculative_algorithm: str | None = None
    sglang_speculative_num_steps: int | None = None
    sglang_speculative_eagle_topk: int | None = None
    sglang_speculative_num_draft_tokens: int | None = None
    sglang_mamba_scheduler_strategy: str = "extra_buffer"
    mtp_num_layers: int | None = None
    enable_mtp_training: bool = False
    mtp_loss_scaling_factor: float | None = None
    sglang_max_running_requests: int | None = 256

    sglang_enable_hierarchical_cache = True
    sglang_hicache_ratio = 1.0
    sglang_hicache_write_policy = "write_through"
    sglang_page_size = 64

    # ── Parallelism ───────────────────────────────────────────────────────
    tensor_model_parallel_size: int = 1
    sequence_parallel: bool = True
    pipeline_model_parallel_size: int = 2
    context_parallel_size: int = 4
    expert_model_parallel_size: int = 4
    expert_tensor_parallel_size: int = 1

    # ── Training ──────────────────────────────────────────────────────────
    n_samples_per_prompt: int = 8
    lr: float = 1e-6
    max_tokens_per_gpu: int = MAX_TOKENS_PER_GPU
    calculate_per_token_loss: bool = True
    moe_token_dispatcher_type: str = "flex"
    moe_enable_deepep: bool = True

    # ── Optimizer ─────────────────────────────────────────────────────────
    optimizer_cpu_offload: bool = True
    overlap_cpu_optimizer_d2h_h2d: bool = True
    use_precision_aware_optimizer: bool = True

    # ── Attention ─────────────────────────────────────────────────────────
    attention_backend: str = "flash"
    no_save_optim: bool = True
    no_load_optim: bool = True
    over_sampling_batch_size: int | None = None
    dynamic_sampling_filter_path: str | None = None

    # ── Checkpointing / eval ──────────────────────────────────────────────
    megatron_to_hf_mode: str = ""
    ref_load: str = ""
    save_interval: int = 5
    eval_interval: int | None = 5

    # ── Chat template ─────────────────────────────────────────────────────
    apply_chat_template_kwargs: dict | str = field(
        default_factory=lambda: {"enable_thinking": True}
    )

    # ── GDPO: dual-clip + per-dimension advantage ─────────────────────────
    #
    # eps_clip_c > 1.0 enables GDPO's dual-clip: when the advantage is
    # negative, the surrogate loss is clipped at -eps_clip_c * advantage,
    # preventing overly pessimistic updates.
    eps_clip_c: float = 3.0

    extra_config: dict | None = field(
        default_factory=lambda: {
            "rl_parallel_generation_tasks": 64,
            # GDPO advantage function: per-dimension GRPO normalization.
            "custom_advantage_function_path": (
                "modal_training_gym.frameworks.slime.gdpo.gdpo_compute_advantages"
            ),
            # Length penalty parameters for the advantage function.
            "gdpo_length_penalty_free_tokens": LENGTH_PENALTY_FREE_TOKENS,
            "gdpo_length_penalty_max_tokens": LENGTH_PENALTY_MAX_TOKENS,
            "gdpo_length_penalty_max_cost": LENGTH_PENALTY_MAX_COST,
        }
    )

    # ── Environment ───────────────────────────────────────────────────────
    environment: dict = field(
        default_factory=lambda: {
            "PYTHONPATH": "/root/Megatron-LM/",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "NCCL_NVLS_ENABLE": "1",
        }
    )

    # ── Custom functions (GDPO overrides) ─────────────────────────────────
    custom_generate_function: Any = toolathlon_gdpo_generate
    custom_rm_function: Any = gdpo_reward_func
    image_overlay: Any = _gdpo_image_overlay
    eval_fn: Any = toolathlon_eval_fn


# ─────────────────────────────────────────────────────────────────────────────
# Entrypoints
# ─────────────────────────────────────────────────────────────────────────────


@app.local_entrypoint()
def train(experiment_name: str, eval_interval: int = 5) -> None:
    try:
        modal.Secret.from_name("huggingface-secret").hydrate()
    except modal.exception.NotFoundError as exc:
        raise RuntimeError(
            "Missing Modal Secret 'huggingface-secret' (needs HF_TOKEN)."
        ) from exc

    recipe = ToolathlonGDPOQwen3_6_35bRecipe(
        wandb=WandbConfig(project="toolathlon-gdpo", group="qwen3.6-35b-a3b"),
        custom_rollout_log_function=gdpo_rollout_log,
        eval_interval=eval_interval or None,
    )
    bad_rollout_shape = (
        recipe.global_batch_size > 64
        or recipe.n_samples_per_prompt > 8
        or recipe.rollout_batch_size > 8
    )
    if bad_rollout_shape:
        raise RuntimeError(
            "Toolathlon live rollout config is too large: "
            f"global_batch_size={recipe.global_batch_size}, "
            f"n_samples_per_prompt={recipe.n_samples_per_prompt}, "
            f"rollout_batch_size={recipe.rollout_batch_size}. "
            "Expected <=64, <=8, <=8 for live Mini-SWE rollouts."
        )
    print(
        "Toolathlon GDPO effective config: "
        f"global_batch_size={recipe.global_batch_size}, "
        f"n_samples_per_prompt={recipe.n_samples_per_prompt}, "
        f"rollout_batch_size={recipe.rollout_batch_size}, "
        f"rollout_max_response_len={recipe.rollout_max_response_len}, "
        f"rollout_max_context_len={recipe.rollout_max_context_len}, "
        f"max_tokens_per_gpu={recipe.max_tokens_per_gpu}, "
        f"eps_clip_c={recipe.eps_clip_c}, "
        f"extra_config={recipe.extra_config}",
        flush=True,
    )

    group = TrainingGroup(
        name=experiment_name,
        base=TrainConfig(
            model=Qwen3_6_35B(),
            dataset=ToolathlonDataset(),
            recipe=recipe,
        ),
        merge_model_recipe=False,
        grid={
            "recipe.sglang_disable_custom_all_reduce": [False],
        },
    )

    print(f"Launching toolathlon GDPO trainings: {group.get_train_configs()}")
    launches = group.launch(prepare_inputs=True)
    print(f"Launched {len(launches)} runs")
    for launch in launches:
        print(
            f"  {launch.training_run_id}  "
            f"(app_id={launch.modal_app_id}, call_id={launch.function_call_id})"
        )
    if group.failures:
        for overrides, err in group.failures:
            print(f"  FAILED {overrides}: {_exception_summary(err)}")
    return None


@app.local_entrypoint()
def ablation(experiment_name: str = "gdpo-ablation") -> None:
    """Launch parallel ablation arms to diagnose pass-rate collapse.

    Arms (all share lr=3e-7 to fix catastrophic forgetting; ablate length penalty):
      1. low-lr:         lr=3e-7, full GDPO (max_cost=0.25)
      2. low-lr-weak:    lr=3e-7, weaker penalty (max_cost=0.1)
      3. low-lr-no-pen:  lr=3e-7, no length penalty (max_cost=0.0, = pure GRPO)
    """
    try:
        modal.Secret.from_name("huggingface-secret").hydrate()
    except modal.exception.NotFoundError as exc:
        raise RuntimeError(
            "Missing Modal Secret 'huggingface-secret' (needs HF_TOKEN)."
        ) from exc

    arms: list[tuple[str, dict]] = [
        # arm 1: lower LR, same GDPO penalty
        (
            "low-lr",
            {"lr": 3e-7, "max_cost": 0.25, "free_tokens": 4000, "max_tokens": 16000},
        ),
        # arm 2: lower LR + weaker penalty
        (
            "low-lr-weak",
            {"lr": 3e-7, "max_cost": 0.1, "free_tokens": 6000, "max_tokens": 16000},
        ),
        # arm 3: lower LR, no length penalty (pure task-reward GRPO)
        (
            "low-lr-no-pen",
            {"lr": 3e-7, "max_cost": 0.0, "free_tokens": 4000, "max_tokens": 16000},
        ),
    ]

    all_launches = []
    all_failures = []

    for arm_name, params in arms:
        extra = {
            "rl_parallel_generation_tasks": 64,
            "custom_advantage_function_path": (
                "modal_training_gym.frameworks.slime.gdpo.gdpo_compute_advantages"
            ),
            "gdpo_length_penalty_free_tokens": params["free_tokens"],
            "gdpo_length_penalty_max_tokens": params["max_tokens"],
            "gdpo_length_penalty_max_cost": params["max_cost"],
        }
        recipe = ToolathlonGDPOQwen3_6_35bRecipe(
            lr=params["lr"],
            extra_config=extra,
            wandb=WandbConfig(
                project="toolathlon-gdpo",
                group=f"ablation-{arm_name}",
            ),
            custom_rollout_log_function=gdpo_rollout_log,
            eval_interval=None,
            save_interval=5,
        )
        group = TrainingGroup(
            name=f"{experiment_name}-{arm_name}",
            base=TrainConfig(
                model=Qwen3_6_35B(),
                dataset=ToolathlonDataset(),
                recipe=recipe,
            ),
            merge_model_recipe=False,
            grid={"recipe.sglang_disable_custom_all_reduce": [False]},
        )
        print(f"\n{'=' * 60}")
        print(f"Launching arm: {arm_name}")
        print(
            f"  lr={params['lr']}, max_cost={params['max_cost']}, "
            f"free_tokens={params['free_tokens']}"
        )
        print(f"{'=' * 60}")
        launches = group.launch(prepare_inputs=True)
        all_launches.extend(launches)
        all_failures.extend(group.failures)

    print(f"\n{'=' * 60}")
    print(f"ABLATION SUMMARY: {len(all_launches)} launched, {len(all_failures)} failed")
    for launch in all_launches:
        print(f"  {launch.training_run_id}  (app_id={launch.modal_app_id})")
    for overrides, err in all_failures:
        print(f"  FAILED {overrides}: {_exception_summary(err)}")
    return None


@app.local_entrypoint()
def eval(
    training_run_id: str = "",
    max_concurrency: int = 4,
    step_limit: int = MINI_SWE_STEP_LIMIT,
    max_new_tokens: int = ROLLOUT_RESPONSE_LEN,
    mini_swe_command_timeout_sec: int = 120,
    thinking_budget: int = MINI_SWE_THINKING_BUDGET,
    obs_char_limit: int = 16000,
    eval_fn_path: str = "",
) -> None:
    """Evaluate a deployed model on the Tier-A Toolathlon tasks."""
    from modal_training_gym import DeploymentConfig, list_checkpoints
    from modal_training_gym.common.eval import EvalConfig
    from modal_training_gym.deploy_recipes.sglang_recipe import SglangRecipe

    def _status(state: str, detail: str = "") -> None:
        print(f"\n=== [eval] {state}{f' — {detail}' if detail else ''} ===", flush=True)

    model = Qwen3_6_35B()
    serve_recipe = SglangRecipe(
        gpu="H200", tp=1, extra_server_args={"--trust-remote-code": ""}
    )
    checkpoint = list_checkpoints(training_run_id)[-1] if training_run_id else None

    _status(
        "DEPLOYING MODEL",
        checkpoint.path if checkpoint is not None else model.model_name,
    )
    try:
        deployment = DeploymentConfig(
            model=model,
            checkpoint=checkpoint,
            recipe=serve_recipe,
            app_name="toolathlon-gdpo-serve",
            served_model_name="toolathlon-gdpo",
        ).serve()
    except Exception as exc:  # noqa: BLE001
        _status("FAILED", f"deploy: {exc}")
        raise
    print(f"Deployed to {deployment.url}")

    if eval_fn_path:
        import importlib

        module_path, _, attr = eval_fn_path.rpartition(".")
        eval_fn = getattr(importlib.import_module(module_path), attr)
    else:
        eval_fn = make_toolathlon_eval_fn(
            sampling_params={
                "temperature": TEMPERATURE,
                "top_p": 1.0,
                "max_new_tokens": max_new_tokens,
            },
            step_limit=step_limit,
            max_new_tokens=max_new_tokens,
            mini_swe_command_timeout_sec=mini_swe_command_timeout_sec,
            thinking_budget=thinking_budget,
            obs_char_limit=obs_char_limit,
        )

    eval_config = EvalConfig(
        dataset=ToolathlonDataset(),
        eval_fn=eval_fn,
        prompt_column="prompt",
    )

    _status("RUNNING EVAL")
    try:
        result = eval_config.evaluate(
            deployment, debug=True, max_concurrency=max_concurrency
        )
    except Exception as exc:  # noqa: BLE001
        _status("FAILED", f"eval: {exc}")
        raise
    _status("SUCCESS", f"pass rate {result.mean:.3f} over {result.total} tasks")
