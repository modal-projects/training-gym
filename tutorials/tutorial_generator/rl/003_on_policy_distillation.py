# pyright: reportUndefinedVariable=false, reportMissingImports=false
"""Tutorial source for `004_on_policy_distillation` — parsed by generate_tutorial.py."""

TUTORIAL_METADATA = {
    "framework": "`slime`",
    "cluster_shape": "1 × 2×H100",
    "summary": "On-policy distillation on math — Qwen3-8B teacher, Qwen3-4B student",
    "difficulty": "Intermediate",
    "order": 30,
    "api_classes": [
        "Qwen3_4B",
        "Qwen3_8B",
        "DeploymentConfig",
        "EvalConfig",
        "EvalRowResult",
        "SlimeRecipe",
        "TrainConfig",
    ],
}

from tutorial_generator import code, markdown, notebook_only, py_only, shell


@markdown
def _intro():
    """
    # On-policy distillation: Teacher model trains a student model

    In this tutorial, we take two models, Qwen3-8B and Qwen3-4B, and use 
    the larger 8B model to teach the smaller 4B model on its generation logprobs.

    Using the Slime framework and Modal Training Gym, we can easily self-host the
    teacher model on a H100 machine, hit the /generate endpoint with "return_logprob=True",
    and penalize the student model's logprobs per token in input sequence using reverse KL divergence.

    The tutorial follows these steps:

    1. **Deploy the teacher** (Qwen3-8B) on an SGLang server with `DeploymentConfig`.
    2. **Load a math dataset** (`dapo-math-17k`) and define a verifiable eval that checks `Answer: \\boxed{N}`.
    3. **Evaluate the base student** (Qwen3-4B) to get a baseline accuracy.
    4. **Define a reward function** that calls the teacher's `/generate` endpoint with `return_logprob=True` and combines the teacher log-probs with a math correctness score.
    5. **Train with GRPO + OPD** using `SlimeRecipe` — slime applies a per-token reverse KL penalty from the teacher log-probs on top of the GRPO advantage.
    6. **Evaluate the trained student** and compare accuracy before vs after.

    ### How OPD works

    During each rollout step:
    1. The student generates a response (math reasoning).
    2. The student's token IDs are sent to the teacher's SGLang server.
    3. The teacher returns per-token log-probabilities.
    4. Slime modifies the advantage at each token:

    $$A_t = A_t^{\\text{GRPO}} - \\lambda_{\\text{opd}} \\cdot (\\log \\pi_{\\text{student}} - \\log \\pi_{\\text{teacher}})$$

    The first term pushes toward correct math answers (sparse reward).
    The second term pushes toward the teacher's token-level distribution
    (dense signal at every position). Together they teach the student
    *what* to say (correct answer) and *how* to say it (teacher-like
    reasoning).

    ### Dataset

    We use [`zhuzilin/dapo-math-17k`](https://huggingface.co/datasets/zhuzilin/dapo-math-17k),
    the same math dataset used by slime's own OPD examples. Each row is a math
    problem with a ground-truth integer answer. The model is prompted to
    respond with `Answer: \\boxed{N}` — evaluation just checks whether the
    number matches.
    """


@py_only
@markdown
def _run_instructions():
    """
    Run with:
    ```
    uv run python tutorials/rl/003_on_policy_distillation/003_on_policy_distillation.py
    ```
    """


@notebook_only
@shell("%uv pip install -q git+https://github.com/modal-projects/training-gym.git@main")
def _install():
    pass


@code
def _imports():
    import re

    from modal_training_gym import (
        DeploymentConfig,
        EvalConfig,
        EvalRowResult,
        HuggingFaceDataset,
        ModelDeployment,
        Qwen3_4B,
        Qwen3_8B,
        SlimeRecipe,
        TrainConfig,
        list_checkpoints,
    )
    from modal_training_gym.deploy_recipes.sglang_recipe import SglangRecipe


# ── Deploy teacher ───────────────────────────────────────────────────────


@markdown
def _deploy_intro():
    """
    ## Deploy the teacher model

    We borrow a recipe from Modal's Training Gym repo to deploy our teacher model on an SGLang server.
    """


@code
def _deploy_teacher():
    teacher_model = Qwen3_8B()
    teacher_deployment = DeploymentConfig(
        model=teacher_model,
        recipe=SglangRecipe(gpu="H100"),
        app_name="opd-teacher-qwen3-8b",
        served_model_name="qwen3-8b-teacher",
    ).serve()
    print(f"Teacher URL: {teacher_deployment.url}")

    TEACHER_GENERATE_URL = f"{teacher_deployment.url}/generate"


@notebook_only
@markdown
def _teacher_test_intro():
    """
    Is our teacher good at answering math problems? For a 8B model, it may not be the best
    but will surely outperform our smaller, 4B model by a large margin. Every parameter counts!
    """


@notebook_only
@code
def _teacher_test():
    response = teacher_deployment.generate(
        "Solve the following math problem step by step. The last line of your "
        "response should be of the form Answer: \\boxed{$Answer} where $Answer "
        "is the answer to the problem.\n\nWhat is 17 * 23?",
        chat_template_kwargs={"enable_thinking": True},
    )
    print(response[-200:])


# ── Dataset ──────────────────────────────────────────────────────────────


@markdown
def _dataset_intro():
    """
    ## Dataset

    We use a simple math dataset containing competition problems that the LLM is tasked
    with answering via a `Answer: \\boxed{N}` response. This simple format allows
    for determnistic evaluation!

    [Here's the link to `zhuzilin/dapo-math-17k`](https://huggingface.co/datasets/zhuzilin/dapo-math-17k)
    
    Each row contains a `prompt` field with the original chat message and a `label` containing an integer answer.

    [Thinking Machines](https://thinkingmachines.ai/blog/on-policy-distillation/) demonstrates that 
    using a small number of samples with a larger number of rollouts can be sufficient for OPD.
    For this tutorial, we take 100 training samples and hold out 20 for evaluation.
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
    rows = eval_dataset.load()
    for row in rows[:2]:
        prompt = row["prompt"]
        if isinstance(prompt, list):
            prompt = prompt[0]["content"] if prompt else ""
        print(prompt[:200])
        print(f"  label: {row['label']}")
        print()


# ── Math evaluation ──────────────────────────────────────────────────────


@code
def _eval_fn():
    def _normalize_answer(answer: str) -> str:
        answer = str(answer).strip()
        answer = answer.split("=")[-1]
        for old, new in [("$", ""), ("\\$", ""), (",", ""), (" ", ""),
                          ("\\text{", ""), ("}", ""), ("\\boxed{", "")]:
            answer = answer.replace(old, new)
        return answer.strip()

    def _extract_answer(response: str) -> str:
        match = re.findall(r"(?i)Answer\s*:\s*([^\n]+)", response)
        return match[-1].strip() if match else "[INVALID]"

    def _check_math(response: str, label: str) -> bool:
        pred = _normalize_answer(_extract_answer(response))
        gt = _normalize_answer(label)
        try:
            gt = str(int(float(gt)))
        except (ValueError, OverflowError):
            pass
        return pred == gt

    def math_eval_fn(deployment: ModelDeployment, example: dict) -> EvalRowResult:
        prompt = example.get("prompt", "")
        if isinstance(prompt, list):
            prompt = prompt[0]["content"] if prompt else ""
        label = example.get("label", "")

        response = deployment.generate(
            prompt,
            ensure_ready=False,
            chat_template_kwargs={"enable_thinking": True},
        )

        correct = _check_math(response, label)
        pred = _normalize_answer(_extract_answer(response))

        return EvalRowResult(
            score=1.0 if correct else 0.0,
            response=response,
            metadata={"correct": correct, "pred": pred, "label": label},
        )


# ── Evaluate base student ────────────────────────────────────────────────


@markdown
def _eval_base_intro():
    """
    ## Baseline Eval

    Let's run the math eval on our base serving model. Thankfully, our dataset requires
    simple-enough answers that a tiny, 4B model should not cause issues for our deterministic parser.
    In our own experience, requiring a strict JSON output format can cause evaluation issues!
    See [this LoRA adapter for making Qwen3-4B successful at structured output](https://huggingface.co/uchkw/qwen3-4b-structured-output-lora). 
    """


@code
def _eval_base():
    base_model = Qwen3_4B()
    base_deployment = DeploymentConfig(model=base_model).serve()
    print(f"Student URL: {base_deployment.url}")

    eval_config = EvalConfig(dataset=eval_dataset, eval_fn=math_eval_fn)
    print("--- Evaluating base student... ---")
    base_eval = eval_config.evaluate(base_deployment, debug=True)
    n_correct = sum(1 for r in base_eval.rows if r.metadata.get("correct"))
    print(f"Base accuracy: {n_correct}/{len(base_eval.rows)} "
          f"({base_eval.mean:.1%})")


@notebook_only
@code
def _base_examples():
    for r in base_eval.rows[:3]:
        status = "CORRECT" if r.metadata["correct"] else "WRONG"
        print(f"[{status}] label={r.metadata['label']}, pred={r.metadata['pred']}")
        print(f"  ...{r.response[-150:]}")
        print()


# ── Custom reward for OPD + math ─────────────────────────────────────────


@markdown
def _rm_intro():
    """
    ## Reward function

    OPD uses "reverse" KL divergence to grade the student model's output. Remember,
    KL divergence D_kl(P || Q) is Sigma_x P(x) * log ()P(x) / Q(x)), where P is ground-truth.
    While the student is supposed to be learning off of the Teacher, we use the "reverse"
    KL divergence where P is the student's probability distribution, allowing us to apply
    a teacher penalty to modes only valued by the student model. In the "reverse" case,
    where P is the teacher's probability distribution, we apply a penalty to modes that 
    the teacher thinks is valuable but may be out of distribution for the student model.

    Additionally, we apply a determnistic reward signal from the evaluation function to
    better improve our model on producing the right answer. By using Modal Training Gym
    with Slime, we can specify a custom reward function that wraps the out-of-the-box 
    for Slime and combines the reverse KL divergence with a binary integer reward signal.
    """


@code
def _rm():
    async def math_opd_rm(args, sample, **kwargs):
        """Collect teacher log-probs AND compute math correctness reward.

        The teacher log-prob collection is handled by slime's built-in OPD
        reward function. We call it, then add our scalar math reward.
        """
        from slime.rollout.on_policy_distillation import reward_func as _opd_reward

        teacher_response = await _opd_reward(args, sample, **kwargs)

        label = getattr(sample, "label", "") or ""
        correct = _check_math(sample.response, label)
        sample.math_correct = correct

        return teacher_response

    def math_opd_post_process(args, samples, **kwargs):
        """Post-process: store teacher log-probs and return math rewards.

        Delegates to slime's built-in post_process_rewards for the teacher
        log-prob alignment, then overrides the scalar rewards with math
        correctness scores.
        """
        from slime.rollout.on_policy_distillation import post_process_rewards as _opd_post

        _, _ = _opd_post(args, samples, **kwargs)

        math_rewards = []
        for sample in samples:
            correct = getattr(sample, "math_correct", False)
            math_rewards.append(1.0 if correct else -1.0)

        return math_rewards, math_rewards


# ── Training ─────────────────────────────────────────────────────────────


@markdown
def _train_intro():
    """
    ## Training

    The training recipe uses 1 H100 GPU per actor and rollout engine. The actor engine
    runs the training and the rollout engine runs the model for inference/forward passes.
    You may want to tune the batch size for fitting the memory requirements of your GPU
    and increase the samples per prompt parameter for generating more variants per group.
    The total_rollouts_per_step is the rollout_batch_size * n_samples_per_prompt, and
    the total # of rollouts that occur over a training run is the total_rollouts_per_step * num_rollout.
    """


@code
def _train():
    training_run = TrainConfig(
        model=base_model,
        dataset=train_dataset,
        recipe=SlimeRecipe(
            custom_rm_function=math_opd_rm,

            gpu_type="H100",
            colocate=True,
            actor_num_gpus_per_node=1,
            rollout_num_gpus=1,
            tensor_model_parallel_size=1,
            sequence_parallel=False,
            rollout_num_gpus_per_engine=1,

            num_rollout=10,
            rollout_batch_size=16,
            n_samples_per_prompt=4,
            rollout_max_response_len=2048,
            rollout_temperature=1.0,

            global_batch_size=16,
            lr=1e-6,
            save_interval=5,
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

    print("--- Starting OPD training... ---")
    print(f"  Teacher: {teacher_deployment.url}")
    print(f"  Student: Qwen3-4B")
    print(f"  Dataset: dapo-math-17k (100 problems)")
    train_result = training_run.train()
    print(f"Training run id: {train_result.training_run_id}")
    print("--- Training complete ---")


# ── Evaluate trained ─────────────────────────────────────────────────────


@markdown
def _eval_trained_intro():
    """
    ## Evaluate the trained student

    Let's run the evaluation on our trained model and compare it
    to the baseline evaluation from earlier. 
    """


@code
def _eval_trained():
    checkpoint = list_checkpoints(train_result.training_run_id)[-1]
    print(f"Checkpoint: {checkpoint.path}")

    trained_deployment = DeploymentConfig(
        model=Qwen3_4B(),
        checkpoint=checkpoint,
        app_name="qwen3-4b-opd-trained-serve",
        served_model_name="qwen3-4b-opd",
    ).serve()
    print(f"Trained student URL: {trained_deployment.url}")

    print("--- Evaluating trained student... ---")
    trained_eval = eval_config.evaluate(trained_deployment, debug=True)
    n_correct = sum(1 for r in trained_eval.rows if r.metadata.get("correct"))
    print(f"Trained accuracy: {n_correct}/{len(trained_eval.rows)} "
          f"({trained_eval.mean:.1%})")

@notebook_only
@code
def _trained_examples():
    for base_r, trained_r in zip(base_eval.rows[:3], trained_eval.rows[:3]):
        label = base_r.metadata["label"]
        b_status = "CORRECT" if base_r.metadata["correct"] else "WRONG"
        t_status = "CORRECT" if trained_r.metadata["correct"] else "WRONG"
        print(f"label={label}")
        print(f"  Base:    [{b_status}] pred={base_r.metadata['pred']}")
        print(f"  Trained: [{t_status}] pred={trained_r.metadata['pred']}")
        print()


@markdown
def _compare_intro():
    """
    ## Results

    Let's hope you see a positive delta on your eval performance!
    """


@code
def _compare():
    base_correct = sum(1 for r in base_eval.rows if r.metadata.get("correct"))
    trained_correct = sum(1 for r in trained_eval.rows if r.metadata.get("correct"))
    total = len(base_eval.rows)
    print(f"Base student:    {base_correct}/{total} ({base_eval.mean:.1%})")
    print(f"Trained student: {trained_correct}/{total} ({trained_eval.mean:.1%})")
    print(f"Delta:           {trained_eval.mean - base_eval.mean:+.1%}")


@markdown
def _next_steps():
    """
    ## Next steps
    
    Some cool ways to extend and improve this example:
    1. Use a bigger teacher: Qwen3 offers models in the 32B paramter range. 
    This model will fit on a 4xH100 GPU setup and can show measurable improvements
    on the student model evaluation delta.
    2. Tweak the composite reward signal: Try applying a coefficient like *2* to the
    binary integer reward signal used in the custom reward function to value correct answers
    over student-teacher alignment.
    3. Try cross-family distillation: Use a teacher from a different model family (e.g. Kimi K2)
    to train our Qwen3-4B student model. You may run into cross-tokenizer differences, so
    be careful to only grade logprobs on tokens that exist in both models vocabularies and
    align 1:1 on a per-character basis. 
    """