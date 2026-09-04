# ---
# order: 4
# ---
#
# # Efficient inference using on-policy distillation
#
# For workloads where you need the throughput and/or cost-savings of a smaller model,
# but the capabilities afforded by a larger model, on-policy distillation (OPD) is
# an effective and practical way to hit your targets. In this tutorial, we'll use
# [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) to teach the smaller 
# [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) how to better solve 
# olympiad-style math problems sourced from the
# [zhuzilin/dapo-math-17k](https://huggingface.co/datasets/zhuzilin/dapo-math-17k)
# Huggingface dataset.
#
# <details>
# <summary>How does OPD work?</summary>
#
# In addition to the standard RL algorithm, during each rollout step,
# the student's response token IDs are sent to the teacher which returns
# per-token log-probabilities. This changes the advantage, the extra signal each token
# gets based on how much better or worse the response was than expected, as follows:
#
# $$
# A_t = A_t^{\text{GRPO}} - \lambda_{\text{opd}} \cdot (\log \pi_{\text{student}} - \log \pi_{\text{teacher}})
# $$
#
# The first term represents sparse rewards for correct answers,
# while the second term represents the dense signals that push towards the
# teacher's token-level distribution. Together, they teach the student
# what to say (i.e., getting the correct answer) and how to say it (i.e.,
# how the teacher would respond).
#
# </details>
#
# To do cross-family OPD (i.e., use a teacher from a different model family such as Deepseek), see
# [this tutorial](https://gym.modal.dev/tutorials/cross_tokenizer_distillation).

import re

from modal_training_gym import (
    CustomDeployment,
    Endpoint,
    HuggingFaceDataset,
    Qwen3_5_4B,
    Qwen3_5_4B_Recipe,
    Qwen3_5_9B,
    TrainConfig,
)

# ## Deploy the base models
#
# First, we'll deploy the teacher and base models to derive a baseline.
# We can use an [Endpoint](https://modal.com/docs/guide/endpoints)
# to serve the student. However, for the teacher model, we need per-token logprobs, 
# which are not currently supported by Endpoints when speculative decoding is
# enabled. So we instead use a
# [CustomDeployment](https://gym.modal.dev/reference/customdeployment)
# to serve the teacher.

student_model = Qwen3_5_4B()

base_student_deployment = Endpoint.launch(
    student_model, unauthenticated=True, recreate_if_existing=True
)

teacher_model = Qwen3_5_9B()
teacher_deployment = CustomDeployment.launch(
    teacher_model,
    app_name="qwen3.5-9b-teacher",
    unauthenticated=True,
)

base_student_deployment.wait_until_ready(timeout=15 * 60)
print(f"student base model deployed to {base_student_deployment.url}")

teacher_deployment.wait_until_ready(timeout=15 * 60)
print(f"teacher base model deployed to {teacher_deployment.url}")

TEACHER_GENERATE_URL = f"{teacher_deployment.url}/generate"

# ## Define a scoring function
#
# Following the [DAPO paper](https://arxiv.org/abs/2503.14476), we'll normalize as 
# they do and return 1 for correct answers and -1 for incorrect answers. Although
# we'd like to give a more granular score for predictions, we can't simply use
# the numerical difference between a prediction and a ground-truth answer, since
# a numerically-close answer can be more wrong than one further away.

def _extract_answer(response: str) -> str:
    match = re.findall(r"(?i)Answer\s*:\s*([^\n]+)", response)
    return match[-1].strip() if match else "[INVALID]"

def _normalize_answer(answer: str) -> str:
    answer = str(answer).strip()
    answer = answer.split("=")[-1]
    for old, new in [("$", ""), ("\\$", ""), (",", ""), (" ", ""),
                      ("\\text{", ""), ("}", ""), ("\\boxed{", "")]:
        answer = answer.replace(old, new)
    return answer.strip()

def score_answer(response: str, label: str) -> int:
    pred = _normalize_answer(_extract_answer(response))
    gt = _normalize_answer(label)
    try:
        gt = str(int(float(gt)))
    except (ValueError, OverflowError):
        pass
    return 1 if pred == gt else -1

# ## Get the dataset
#
# As [this Thinking Machines blog](https://thinkingmachines.ai/blog/on-policy-distillation/)
# describes, using a small number of samples with a larger number of rollouts can be
# sufficient for OPD. Following suit, we'll only use 100 training samples and 20 for evaluation.

train_dataset = HuggingFaceDataset(
    "zhuzilin/dapo-math-17k",
    hf_split="train[:100]",
    input_column="prompt",
    output_column="label",
    input_format="messages",
    always_download=True,
)

eval_dataset = HuggingFaceDataset(
    "zhuzilin/dapo-math-17k",
    hf_split="train[100:120]",
    input_column="prompt",
    output_column="label",
    input_format="messages",
    always_download=True,
)

# ## Evaluate the base models
#
# First, we should check if our teacher is good enough to, well, be a teacher.
# Then, we'll see how the student fares in comparison to establish the gap we
# must close.
#
# <details>
# <summary>On strict formats for evaluation</summary>
#
# Thankfully, our dataset requires simple-enough answers that a tiny, 
# 4B model shouldn't cause issues for our deterministic parser. In our own experience,
# requiring a strict JSON output format can cause evaluation issues!
# See [this LoRA adapter](https://huggingface.co/uchkw/qwen3-4b-structured-output-lora)
# for an example of adapting a small Qwen model to strict output formats.
#
# </details>

def run_eval(
    deployment, *, max_concurrency: int = 2
) -> float:
    from concurrent.futures import ThreadPoolExecutor

    deployment.wait_until_ready(timeout=15 * 60)

    def _score_one(example):
        msg = deployment.chat(
            example[eval_dataset.input_key()],
            chat_template_kwargs={"enable_thinking": True},
        )
        response = msg.get("content") or msg.get("reasoning_content") or ""
        return score_answer(response, example[eval_dataset.label_key()])

    with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        scores = list(executor.map(_score_one, eval_dataset.rows()))
    percent_correct = (
        len([s for s in scores if s == 1]) / len(scores) if scores else float("nan")
    )
    return percent_correct

print("running teacher base model evaluation...")
teacher_correct = run_eval(teacher_deployment)
print(f"percent correct: {teacher_correct:.1%}")

print("running student base model evaluation...")
base_student_correct = run_eval(base_student_deployment)
print(f"percent correct: {base_student_correct:.1%}")

# ## Creating a reward function
#
# To use our scoring function for training, we have to modify some framework-specific
# functions. Luckily, this is relatively painless.
#
# <details>
# <summary>How OPD defines KL</summary>
#
# KL-divergence is defined as:
#
# $$
# D_{\mathrm{KL}}(P \| Q) = \sum_x P(x) \log \frac{P(x)}{Q(x)}
# $$
#
# where $P$ is the behavior distribution and $Q$ is the target distribution.
#
# Forward KL treats the teacher model as $P$ and the student model as $Q$. However, the
# $\log \frac{P(x)}{Q(x)}$ term would then be weighted by the teacher model's probability distribution $P$,
# resulting in high surprisal on modes unfamiliar to the student model.
#
# To counter this, OPD uses "reverse" KL divergence to grade the student model's output:
#
# $$
# D_{\mathrm{KL}}(\pi_{\mathrm{student}} \| \pi_{\mathrm{teacher}})
# $$
#
# where our student model is treated as the behavior distribution and
# the teacher model is our target distribution.
#
# When the teacher has high surprisal on a student mode, the term $\log(P(x)) - log(Q(x))$
# will yield a high positive KL divergence to penalize the student model.
# Now, the student model only gets penalized on modes relevant to itself.
#
# </details>
#
# <details>
# <summary>A fun exercise</summary>
#
# Try tweaking the composite reward signal by applying an integer coefficient to the
# binary integer reward signal used in the custom reward function to value correct answers
# over student-teacher alignment.
#
# </details>

async def math_opd_rm(args, sample, **kwargs):
    from slime.rollout.on_policy_distillation import reward_func as _opd_reward

    teacher_response = await _opd_reward(args, sample, **kwargs)

    response = student_model.parse_response(sample.response)
    score = score_answer(response.content, sample.label)
    sample.score = score
    if not isinstance(getattr(sample, "metadata", None), dict):
        sample.metadata = {}
    sample.metadata["shaped_reward"] = float(score)

    return teacher_response

def math_opd_post_process(args, samples, **kwargs):
    from slime.rollout.on_policy_distillation import post_process_rewards as _opd_post

    _, _ = _opd_post(args, samples, **kwargs)

    math_rewards = [getattr(sample, "score", -1) for sample in samples]
    return math_rewards, math_rewards  # quirk of slime

# ## Start training
#
# In addition to the standard parameters we use for most tutorials,
# we set parameters such as `environment` and `extra_config` to supply
# framework-necessary environment variables and flags.

config = TrainConfig(
    model=student_model,
    dataset=train_dataset,
    recipe=Qwen3_5_4B_Recipe(
        eval_interval=None,
        rollout_num_gpus=8,
        num_rollout=10,
        n_samples_per_prompt=4,
        rollout_max_response_len=2048,
        save_interval=5,
        custom_rm_function=math_opd_rm,
        custom_reward_post_process_function=math_opd_post_process,
        apply_chat_template_kwargs='{"enable_thinking": true}',
        environment={
            "PYTHONPATH": "/root/Megatron-LM/:/root",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "NCCL_NVLS_ENABLE": "1",
        },
        extra_config={
            "use_opd": True,
            "opd_type": "sglang",
            "opd_kl_coef": 1.0,
            "rm_url": TEACHER_GENERATE_URL,
        },
    ),
)

run = config.launch()
print(f"run id: {run.training_run_id}")

# ## Evaluate the trained student
#
# We'll deploy our trained student and compare it
# to our baseline evaluation from earlier.

result = run.result()
checkpoint = result.checkpoints()[-1]
print(f"checkpoint: {checkpoint.path}")

trained_student_deployment = Endpoint.launch(
    student_model, checkpoint, unauthenticated=True, recreate_if_existing=True
)
trained_student_deployment.wait_until_ready(timeout=15 * 60)
print(f"checkpoint deployed to {trained_student_deployment.url}")

print("running student checkpoint evaluation...")
trained_student_correct = run_eval(trained_student_deployment)
print(f"percent correct: {trained_student_correct:.1%}")
