# Qwen3-30B-A3B validation example
#
# Validates that:
# 1. SGLang serves the Qwen3-30B-A3B MoE model correctly (tp=2)
# 2. SlimeRecipe trains the model with GRPO
#
# Qwen3-30B-A3B is a Mixture-of-Experts model:
#   - 30B total params, ~3B active per token
#   - 128 experts, top-8 routing
#   - 48 layers, hidden_size=2048
#
# Run with:
#   MODAL_FORCE_BUILD=1 uv run modal run examples/qwen3_30b_validation.py

import re

import modal

from modal_training_gym import (
    DeploymentConfig,
    EvalConfig,
    EvalRowResult,
    HuggingFaceDataset,
    Qwen3_30B,
    SlimeRecipe,
    TrainConfig,
    list_checkpoints,
)
from modal_training_gym.deploy_recipes.sglang_recipe import SglangRecipe


# ── Haiku scoring (same as 000_rl_basics) ────────────────────────────────────

_cmudict_cache = {}


def _get_cmudict() -> dict:
    if not _cmudict_cache:
        import nltk
        from nltk.corpus import cmudict

        nltk.download("cmudict", quiet=True)
        _cmudict_cache.update(cmudict.dict())
    return _cmudict_cache


def _count_syllables(text: str) -> int:
    cmu = _get_cmudict()
    total = 0
    for word in re.findall(r"[a-zA-Z]+", text):
        phones = cmu.get(word.lower())
        if phones:
            total += sum(p[-1].isdigit() for p in phones[0])
        else:
            count = len(re.findall(r"[aeiouy]+", word.lower()))
            if word.lower().endswith("e") and count > 1:
                count -= 1
            total += max(count, 1)
    return total


def score_haiku(response: str) -> float:
    lines = [line.strip() for line in response.strip().split("\n") if line.strip()]
    if len(lines) != 3:
        return -10
    total_diff = sum(
        abs(_count_syllables(line) - target) for line, target in zip(lines, [5, 7, 5])
    )
    return -float(total_diff)


# ── Dataset ──────────────────────────────────────────────────────────────────


class HaikuDataset(HuggingFaceDataset):
    hf_repo = "statworx/haiku"
    input_column = "keywords"
    output_column = "text"
    output_format = "jsonl"
    apply_chat_template = True
    system_prompt = (
        "You are a haiku poet. Write a haiku about the given topic. "
        "Use the 5-7-5 syllable format across three lines."
    )
    prompt_template = "Write a haiku about {input}."
    always_prepare = True


# ── Eval function ────────────────────────────────────────────────────────────


def eval_response_fn(_example: dict, response: str) -> EvalRowResult:
    return EvalRowResult(score=score_haiku(response), response=response)


# ── Reward model for training ────────────────────────────────────────────────


async def haiku_rm(args, sample, **kwargs) -> float:
    return score_haiku(sample.response)


# ── Main ─────────────────────────────────────────────────────────────────────

tutorial_cli_app = modal.App()


@tutorial_cli_app.local_entrypoint()
def main() -> None:
    try:
        modal.Secret.from_name("huggingface-secret").hydrate()
    except modal.exception.NotFoundError as e:
        raise RuntimeError(
            "Missing Modal Secret 'huggingface-secret'. Create one at "
            "https://modal.com/secrets with an HF_TOKEN entry, then re-run."
        ) from e

    # ── 1. Serve Qwen3-30B-A3B with SGLang (TP=2 for 30B MoE) ───────────
    print("=" * 60)
    print("STEP 1: Deploying Qwen3-30B-A3B with SGLang (tp=2)")
    print("=" * 60)

    base_model = Qwen3_30B()
    base_model_deployment = DeploymentConfig(
        model=base_model,
        recipe=SglangRecipe(tp=2),
    ).serve()
    print(f"Base model deployed to {base_model_deployment.url}")

    # 30B MoE needs extra startup time for CUDA graph compilation (~15 min)
    base_model_deployment.wait_until_ready(timeout=1200)
    print("Base model ready!")

    # ── 2. Evaluate the base model ───────────────────────────────────────
    print("=" * 60)
    print("STEP 2: Evaluating base model on haiku task")
    print("=" * 60)

    train_dataset = HaikuDataset(n_rows=10)
    eval_dataset = HaikuDataset(n_rows=5)

    eval_config = EvalConfig(
        dataset=eval_dataset,
        eval_response_fn=eval_response_fn,
        generate_kwargs={"chat_template_kwargs": {"enable_thinking": False}},
    )

    base_eval = eval_config.evaluate(base_model_deployment, debug=True)
    print(f"Base model average haiku score: {base_eval.mean:.1f}")

    # ── 3. Train with SlimeRecipe (validates GRPO on 30B MoE) ────────────
    #
    # Qwen3-30B-A3B is MoE (128 experts, top-8). We use:
    #   - TP2 + sequence parallelism for the dense attention layers
    #   - rollout_num_gpus_per_engine=2 to match TP
    #   - colocate=True to share GPUs between training and rollout
    #   - Small num_rollout (5) for a quick validation run
    print("=" * 60)
    print("STEP 3: Training Qwen3-30B-A3B with SlimeRecipe (GRPO)")
    print("=" * 60)

    training_run = TrainConfig(
        model=base_model,
        dataset=train_dataset,
        recipe=SlimeRecipe(
            custom_rm_function=haiku_rm,
            gpu_type="H100",
            colocate=True,
            tensor_model_parallel_size=2,
            sequence_parallel=True,
            rollout_num_gpus_per_engine=2,
            num_rollout=5,
            rollout_batch_size=8,
            rollout_max_response_len=4096,
            rollout_temperature=1.0,
            sglang_mem_fraction_static=0.70,
            save_interval=5,
            n_samples_per_prompt=2,
            lr=5e-7,
            max_tokens_per_gpu=4096,
            apply_chat_template_kwargs='{"enable_thinking": false}',
            image_overlay=lambda image: image.run_commands(
                "uv pip install --system aiohttp nltk>=3.8.0",
                "python -c \"import nltk; nltk.download('cmudict', quiet=True)\"",
            ),
        ),
    )

    train_result = training_run.train()
    print(f"Training complete: {train_result.training_run_id}")

    # ── 4. Serve and evaluate trained checkpoint ─────────────────────────
    print("=" * 60)
    print("STEP 4: Deploying and evaluating trained model")
    print("=" * 60)

    checkpoint = list_checkpoints(train_result.training_run_id)[-1]
    print(f"Checkpoint: {checkpoint.path}")

    trained_deployment = DeploymentConfig(
        model=Qwen3_30B(),
        checkpoint=checkpoint,
        recipe=SglangRecipe(tp=2),
        app_name="qwen3-30b-a3b-haiku-serve",
        served_model_name="qwen3-30b-a3b-haiku",
    ).serve()
    print(f"Trained model deployed to {trained_deployment.url}")

    trained_deployment.wait_until_ready(timeout=1200)
    trained_eval = eval_config.evaluate(trained_deployment, debug=True)
    print(f"Trained model average haiku score: {trained_eval.mean:.1f}")

    # ── Summary ──────────────────────────────────────────────────────────
    print("=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print("Model:          Qwen3-30B-A3B (MoE, 128 experts, top-8)")
    print("SGLang config:  tp=2, sglang_image=v0.5.12-cu130")
    print("SlimeRecipe:    tp=2, colocate, GRPO")
    print(f"Base score:     {base_eval.mean:.1f}")
    print(f"Trained score:  {trained_eval.mean:.1f}")
    print(f"Delta:          {trained_eval.mean - base_eval.mean:+.1f}")
