"""Tutorial source for `003_on_policy_distillation` — parsed by generate_tutorial.py."""

TUTORIAL_METADATA = {
    "framework": "`slime`",
    "cluster_shape": "1 × 1×H100 (teacher) + 1 × 8×H100 (train)",
    "summary": "Teacher-student (OPD) distillation",
    "difficulty": "Intermediate",
    "order": 30,
    "api_classes": [
        "Qwen3_5_4B",
        "Qwen3_5_9B",
        "CustomDeployment",
        "Endpoint",
        "SlimeRecipe",
        "TrainConfig",
    ],
}

from tutorial_generator import code, markdown, notebook_only, py_only, shell


@markdown
def _intro():
    r"""
    # Saving parameters with on-policy distillation

    For workloads where you need the throughput and/or cost-savings of a smaller model,
    but the capabilities afforded by a larger model, on-policy distillation (OPD) is
    an effective and practical way to hit your targets. In this tutorial, we'll use
    [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) to teach the smaller 
    [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) how to better solve 
    olympiad-style math problems sourced from the
    [zhuzilin/dapo-math-17k](https://huggingface.co/datasets/zhuzilin/dapo-math-17k)
    Huggingface dataset.

    <details>
    <summary>How does OPD work?</summary>

    In addition to the standard RL algorithm, during each rollout step,
    the student's response token IDs are sent to the teacher which returns
    per-token log-probabilities. This changes the advantage, the extra signal each token
    gets based on how much better or worse the response was than expected, as follows:

    $$
    A_t = A_t^{\text{GRPO}} - \lambda_{\text{opd}} \cdot (\log \pi_{\text{student}} - \log \pi_{\text{teacher}})
    $$

    The first term represents sparse rewards for correct answers,
    while the second term represents the dense signals that push towards the
    teacher's token-level distribution. Together, they teach the student
    what to say (i.e., getting the correct answer) and how to say it (i.e.,
    how the teacher would respond).

    </details>

    To do cross-family OPD (i.e., use a teacher from a different model family such as Deepseek), see
    [this tutorial](https://gym.modal.dev/tutorials/rl/009_cross_tokenizer_distillation/).
    """


@py_only
@markdown
def _run_instructions():
    """
    Run with:
    ```
    uv run tutorials/rl/003_on_policy_distillation/003_on_policy_distillation.py
    ```
    """


@notebook_only
@shell(
    "import importlib.util\n"
    "\n"
    "# Skip if modal_training_gym is already importable (e.g. a local editable\n"
    "# checkout) so your edits keep taking effect and the env stays synced.\n"
    "if importlib.util.find_spec('modal_training_gym') is None:\n"
    "    %uv pip install -q git+https://github.com/modal-projects/training-gym.git@main"
)
def _install():
    pass


@code
def _imports():
    import re

    from modal_training_gym import (
        CustomDeployment,
        Endpoint,
        HuggingFaceDataset,
        Qwen3_5_4B,
        Qwen3_5_9B,
        SlimeRecipe,
        TrainConfig,
        list_checkpoints,
    )
    from modal_training_gym.deploy_recipes.sglang_recipe import SglangRecipe


@markdown
def _deploy_teacher_intro():
    """
    ## Deploy the base models

    First, we'll deploy the teacher and base models to derive a baseline.
    We can use an [Endpoint](https://gym.modal.dev/reference/deployment/endpoint/) to serve the student.
    However, Endpoints currently do not support the return of logprobs when
    speculative decoding is enabled. So we instead use a
    [CustomDeployment](https://gym.modal.dev/reference/deployment/customdeployment/) to serve the teacher.
    """


@code
def _deploy_base():
    base_student_model = Qwen3_5_4B()
    base_student_deployment = Endpoint.launch(
        base_student_model, unauthenticated=True, recreate_if_existing=True
    )

    base_teacher_model = Qwen3_5_9B()
    base_teacher_deployment = CustomDeployment.launch(
        base_teacher_model,
        unauthenticated=True,
    )

    base_student_deployment.wait_until_ready(timeout=15 * 60)
    print(f"student base model deployed to {base_student_deployment.url}")

    base_teacher_deployment.wait_until_ready(timeout=15 * 60)
    print(f"teacher base model deployed to {base_teacher_deployment.url}")

    TEACHER_GENERATE_URL = f"{base_teacher_deployment.url}/generate"


@markdown
def _score_fn_intro():
    r"""
    ## Define a scoring function

    Following the [DAPO paper](https://arxiv.org/abs/2503.14476), we'll normalize as 
    they do and return 1 for correct answers and -1 for incorrect answers. Although
    we'd like to give a more granular reward for predictions, we can't simply use
    the numerical difference between a prediction and a ground-truth answer, since
    a numerically-close answer can be more wrong than one further away.

    <details>
    <summary>How OPD defines KL</summary>

    KL-divergence is defined as:

    $$
    D_{\mathrm{KL}}(P \| Q) = \sum_x P(x) \log \frac{P(x)}{Q(x)}
    $$

    where $P$ is the behavior distribution and $Q$ is the target distribution.

    Forward KL treats the teacher model as $P$ and the student model as $Q$. However, the
    $\log \frac{P(x)}{Q(x)}$ term would then be weighted by the teacher model's probability distribution $P$,
    resulting in high surprisal on modes unfamiliar to the student model.

    To counter this, OPD uses "reverse" KL divergence to grade the student model's output:

    $$
    D_{\mathrm{KL}}(\pi_{\mathrm{student}} \| \pi_{\mathrm{teacher}})
    $$

    where our student model is treated as the behavior distribution and
    the teacher model is our target distribution.

    When the teacher has high surprisal on a student mode, the term $\log(P(x)) - log(Q(x))$
    will yield a high positive KL divergence to penalize the student model.
    Now, the student model only gets penalized on modes relevant to itself.

    </details>

    <details>
    <summary>A fun exercise</summary>

    Try tweaking the composite reward signal by applying an integer coefficient to the
    binary integer reward signal used in the custom reward function to value correct answers
    over student-teacher alignment.
    
    </details>
    """

@code
def _score_fn():
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

    async def math_opd_rm(args, sample, **kwargs):
        from slime.rollout.on_policy_distillation import reward_func as _opd_reward

        teacher_response = await _opd_reward(args, sample, **kwargs)

        label = getattr(sample, "label", "") or ""
        score = score_answer(sample.response, label)
        sample.score = score

        return teacher_response

    def math_opd_post_process(args, samples, **kwargs):
        from slime.rollout.on_policy_distillation import post_process_rewards as _opd_post

        _, _ = _opd_post(args, samples, **kwargs)

        math_rewards = [getattr(sample, "score", -1) for sample in samples]
        return math_rewards, math_rewards  # quirk of slime


@markdown
def _dataset_intro():
    """
    ## Get the dataset

    As [this Thinking Machines blog](https://thinkingmachines.ai/blog/on-policy-distillation/)
    describes, using a small number of samples with a larger number of rollouts can be
    sufficient for OPD. Following suit, we'll only use 100 training samples and 20 for evaluation.
    """


@code
def _dataset():
    class MathDataset(HuggingFaceDataset):
        hf_repo = "zhuzilin/dapo-math-17k"
        input_column = "prompt"
        output_column = "label"
        output_format = "jsonl"
        apply_chat_template = False

    train_dataset = MathDataset(n_rows=100)
    eval_dataset = MathDataset(n_rows=20)


@notebook_only
@code
def _dataset_peek():
    df = eval_dataset.to_pandas()
    print(len(df))
    df.head(5)


@markdown
def _eval_base_intro():
    """
    ## Evaluate the base models

    First, we should check if our teacher is good enough to, well, be a teacher.
    Then, we'll see how the student fares in comparison to establish the gap we
    must close.

    <details>
    <summary>On strict formats for evaluation</summary>
    
    Thankfully, our dataset requires simple-enough answers that a tiny, 
    4B model shouldn't cause issues for our deterministic parser. In our own experience,
    requiring a strict JSON output format can cause evaluation issues!
    See [this LoRA adapter](https://huggingface.co/uchkw/qwen3-4b-structured-output-lora)
    for an example of adapting a small Qwen model to strict output formats.

    </details>
    """

@code
def _eval_base():
    def run_eval(
        deployment, *, max_concurrency: int = 2
    ) -> float:
        from concurrent.futures import ThreadPoolExecutor

        deployment.wait_until_ready(timeout=15 * 60)

        def _score_one(example):
            prompt = example["prompt"][0]["content"]
            msg = deployment.chat(
                [{"role": "user", "content": prompt}],
                chat_template_kwargs={"enable_thinking": True},
            )
            response = msg.get("content") or msg.get("reasoning_content") or ""
            return score_answer(response, example["label"])

        with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
            scores = list(executor.map(_score_one, eval_dataset.load()))
        percent_correct = (
            len([s for s in scores if s == 1]) / len(scores) if scores else float("nan")
        )
        return percent_correct

    print("running teacher base model evaluation...")
    base_teacher_correct = run_eval(base_teacher_deployment)
    print(f"percent correct: {base_teacher_correct:.1%}")

    print("running student base model evaluation...")
    base_student_correct = run_eval(base_student_deployment)
    print(f"percent correct: {base_student_correct:.1%}")



@markdown
def _train_intro():
    """
    ## Start training

    In addition to the standard parameters we use for most tutorials,
    we set parameters such as `environment` and `extra_config` to supply
    framework-necessary environment variables and flags.
    """

@code
def _train():
    train_run = TrainConfig(
        model=base_student_model,
        dataset=train_dataset,
        recipe=SlimeRecipe(
            gpu_type="H100",
            tensor_model_parallel_size=1,
            rollout_num_gpus_per_engine=1,
            sequence_parallel=False,
            colocate=True,
            actor_num_gpus_per_node=8,
            rollout_num_gpus=8,
            num_rollout=10,
            rollout_batch_size=16,
            n_samples_per_prompt=4,
            rollout_max_response_len=2048,
            rollout_temperature=1.0,
            global_batch_size=16,
            lr=1e-6,
            save_interval=5,
            custom_rm_function=math_opd_rm,
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
                "custom_reward_post_process_path": (
                    "003_on_policy_distillation.math_opd_post_process"
                ),
                "rm_url": TEACHER_GENERATE_URL,
            },
        ),
    )

    train_result = train_run.train()
    print(f"run id: {train_result.training_run_id}")


@markdown
def _eval_trained_intro():
    """
    ## Evaluate the trained student

    We'll deploy our trained student and compare it
    to our baseline evaluation from earlier. 
    """


@code
def _eval_trained():
    checkpoint = list_checkpoints(train_result.training_run_id)[-1]
    print(f"checkpoint: {checkpoint.path}")

    trained_student_deployment = Endpoint.launch(
        Qwen3_5_4B(), checkpoint, unauthenticated=True, recreate_if_existing=True
    )
    trained_student_deployment.wait_until_ready(timeout=15 * 60)
    print(f"checkpoint deployed to {trained_student_deployment.url}")

    print("running student checkpoint evaluation...")
    trained_student_correct = run_eval(trained_student_deployment)
    print(f"percent correct: {trained_student_correct:.1%}")
