"""Tutorial source for `003_glm_gsm8k` — parsed by generate_tutorial.py."""

TUTORIAL_METADATA = {
    "framework": "`slime`",
    "cluster_shape": "8 × 8×H100",
    "summary": "GLM-4.7 (355B MoE) on GSM8K math — serve, evaluate, GRPO-train, compare",
    "difficulty": "Advanced",
    "order": 30,
    "api_classes": [
        "GLM_4_7",
        "DeploymentConfig",
        "EvalConfig",
        "EvalRowResult",
        "TrainConfig",
        "SlimeRecipe",
        "TrainResult",
    ],
}


from tutorial_generator import code, markdown, notebook_only, py_only, shell


@markdown
def _intro():
    """
    # Training GLM-4.7 on GSM8K

    This tutorial trains [GLM-4.7](https://huggingface.co/zai-org/GLM-4.7),
    a 355B-parameter Mixture-of-Experts model (32B active per token),
    on grade-school math problems from [GSM8K](https://huggingface.co/datasets/openai/gsm8k).

    GLM-4.7 is a large MoE model with 160 routed experts and top-8 routing.
    The first 3 layers are dense; the remaining 89 use sparse expert routing.
    Training a model this size requires multi-node parallelism:

    1. Serving the base model with SGLang on 8×H100.
    2. Loading the GSM8K dataset and defining a math-answer scorer.
    3. Evaluating the base model.
    4. GRPO-training with [slime](https://github.com/THUDM/slime) using the built-in math reward.
    5. Evaluating the trained checkpoint and comparing.

    **Cluster shape:** 8 nodes × 8 H100 GPUs (64 GPUs total). Training uses
    TP=8, PP=4, CP=2, EP=16 to shard the model across all GPUs.
    """


@py_only
@markdown
def run_instructions():
    """
    To run the tutorial, run the following command:
    ```
    uv run python tutorials/rl/003_glm_gsm8k/003_glm_gsm8k.py
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
        GLM_4_7,
        HuggingFaceDataset,
        SlimeRecipe,
        TrainConfig,
        list_checkpoints,
    )


@markdown
def _serve_base_intro():
    """
    ## Serve the base model

    First, let's deploy GLM-4.7 so we can test it.
    The 355B model needs 8×H100 GPUs for serving with tensor parallelism.
    """


@code
def _serve_base_model():
    base_model = GLM_4_7()
    base_model_deployment = DeploymentConfig(
        model=base_model,
    ).serve()
    print(f"Base model deployed to {base_model_deployment.url}")


@notebook_only
@markdown
def _try_base_model():
    """
    Let's try asking it a math question to see how it responds.
    """


@notebook_only
@code
def _try_base_model_code():
    response = base_model_deployment.generate(
        "What is 24 * 37?",
        chat_template_kwargs={"enable_thinking": False},
    )
    print(response)


@markdown
def _dataset_intro():
    """
    ## Define the GSM8K dataset

    GSM8K contains grade-school math word problems. Each row has a `question`
    and an `answer` that ends with `#### <number>` — the final numerical answer.

    We'll use the answer extraction pattern to score the model: extract the
    number after `####` from the label and compare it to the model's final number.
    """


@code
def _define_dataset():
    class GSM8KDataset(HuggingFaceDataset):
        hf_repo = "openai/gsm8k"
        hf_config = "main"
        input_column = "question"
        output_column = "answer"
        output_format = "jsonl"
        apply_chat_template = True
        system_prompt = (
            "You are a math problem solver. Solve the given math problem "
            "step by step, then give your final answer as a number on the "
            "last line prefixed with ####. For example: #### 42"
        )
        always_prepare = True

    train_dataset = GSM8KDataset(n_rows=50)
    eval_dataset = GSM8KDataset(n_rows=20, hf_split="test")


@notebook_only
@markdown
def _peek_dataset():
    """
    Let's look at a few examples from the dataset.
    """


@notebook_only
@code
def _peek_dataset_code():
    df = eval_dataset.to_pandas()
    print(f"Eval set: {len(df)} rows")
    df.head(3)


@markdown
def _scoring_intro():
    """
    ## Define the scoring function

    GSM8K answers end with `#### <number>`. We extract the number from both
    the label and the model's response, then check if they match.
    """


@code
def _define_scoring():
    def extract_answer(text: str) -> str | None:
        match = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", text)
        if match:
            return match.group(1).replace(",", "").strip()
        numbers = re.findall(r"-?[\d,]+(?:\.\d+)?", text)
        if numbers:
            return numbers[-1].replace(",", "").strip()
        return None

    def gsm8k_score(example: dict, response: str) -> EvalRowResult:
        expected = extract_answer(example.get("answer", ""))
        predicted = extract_answer(response)
        correct = expected is not None and predicted is not None and expected == predicted
        return EvalRowResult(
            score=1.0 if correct else 0.0,
            response=response,
        )


@notebook_only
@markdown
def _eval_base_intro():
    """
    ## Evaluate the base model
    """


@code
def _eval_base():
    eval_config = EvalConfig(
        dataset=eval_dataset,
        eval_response_fn=gsm8k_score,
        generate_kwargs={"chat_template_kwargs": {"enable_thinking": False}},
    )
    print("--- Running base model evaluation... ---")
    base_eval = eval_config.evaluate(base_model_deployment, debug=True)
    print(f"Base model GSM8K accuracy: {base_eval.mean:.1%}")
    print("--- Base model evaluation complete ---")


@markdown
def _train_intro():
    """
    ## Train with slime

    Now let's GRPO-train GLM-4.7 on GSM8K using the built-in `math`
    reward type. This reward extracts the final number from the model's
    response and checks it against the label — the same logic as our eval
    scoring function.

    Because GLM-4.7 is a 355B MoE model, the recipe uses heavy parallelism:
    - **TP=8:** tensor parallelism across all GPUs in a node.
    - **PP=4:** pipeline parallelism across 4 stages.
    - **EP=16:** expert parallelism shards the 160 experts across 16 groups.
    - **CP=2:** context parallelism for long sequences.
    - **CPU optimizer offload:** frees GPU memory for the large parameter count.
    """


@code
def _define_training():
    training_run = TrainConfig(
        model=base_model,
        dataset=train_dataset,
        recipe=SlimeRecipe(
            rm_type="math",

            gpu_type="H100",
            colocate=True,
            actor_num_nodes=8,
            actor_num_gpus_per_node=8,
            tensor_model_parallel_size=8,
            sequence_parallel=True,
            rollout_num_gpus_per_engine=32,

            expert_model_parallel_size=16,
            expert_tensor_parallel_size=1,
            pipeline_model_parallel_size=4,
            context_parallel_size=2,
            attention_backend="flash",

            num_rollout=10,
            rollout_batch_size=64,
            rollout_max_response_len=4096,
            rollout_temperature=1.0,
            sglang_mem_fraction_static=0.70,

            n_samples_per_prompt=8,
            global_batch_size=512,
            max_tokens_per_gpu=16384,

            optimizer_cpu_offload=True,
            overlap_cpu_optimizer_d2h_h2d=True,
            use_precision_aware_optimizer=True,

            save_interval=5,
            apply_chat_template_kwargs='{"enable_thinking": false}',
        ),
    )


@markdown
def _launch_training():
    """
    ## Launch training

    `TrainConfig.train()` builds the Modal app, launches the 64-GPU cluster
    (8 nodes × 8 H100), runs GRPO training, and returns a `TrainResult`.
    """


@code
def _run_training():
    print("--- Running training... ---")
    train_result = training_run.train()
    print("--- Training complete ---")


@markdown
def _eval_trained_intro():
    """
    ## Serve and evaluate the trained checkpoint

    Let's deploy the trained checkpoint and run the same GSM8K eval.
    """


@code
def _serve_trained():
    checkpoint = list_checkpoints(train_result.training_run_id)[-1]
    print(f"Checkpoint: {checkpoint.path}")

    trained_model_deployment = DeploymentConfig(
        model=GLM_4_7(),
        checkpoint=checkpoint,
        app_name="glm-4.7-gsm8k-serve",
        served_model_name="glm-4.7-gsm8k",
    ).serve()
    print(f"Trained model deployed to {trained_model_deployment.url}")


@code
def _eval_trained():
    print("--- Running trained model evaluation... ---")
    trained_eval = eval_config.evaluate(trained_model_deployment, debug=True)
    print(f"Trained model GSM8K accuracy: {trained_eval.mean:.1%}")
    print("--- Trained model evaluation complete ---")


@markdown
def _compare_intro():
    """
    ## Compare results
    """


@code
def _compare():
    print(f"Base model GSM8K accuracy:    {base_eval.mean:.1%}")
    print(f"Trained model GSM8K accuracy: {trained_eval.mean:.1%}")
    improvement = trained_eval.mean - base_eval.mean
    print(f"Improvement: {improvement:+.1%}")
