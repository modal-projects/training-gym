# pyright: reportUndefinedVariable=false, reportMissingImports=false
"""Tutorial source for `002_multiturn_number_guessing` — parsed by generate_tutorial.py."""

TUTORIAL_METADATA = {
    "framework": "`slime`",
    "cluster_shape": "1 × 1×H100",
    "summary": "Multi-turn number-guessing RL with custom generate and reward functions",
    "difficulty": "Intermediate",
    "order": 30,
    "api_classes": [
        "DatasetConfig",
        "DeploymentConfig",
        "EvalConfig",
        "EvalRowResult",
        "ModelDeployment",
        "Qwen3_4B",
        "SlimeRecipe",
        "TrainConfig",
        "WandbConfig",
    ],
}

from tutorial_generator import code, markdown, notebook_only, py_only, shell


@markdown
def _intro():
    """
    # Multi-turn RL: guess a number from 1 to 20

    This tutorial builds a small multi-turn environment inspired by PipelineRL's
    guessing task. The model must guess a hidden number in `[1, 20]`.

    Loop per rollout:
    1. Model emits a guess as `<answer>N</answer>`.
    2. Environment returns `<feedback>higher</feedback>`, `<feedback>lower</feedback>`,
       or success.
    3. Repeat for a fixed turn budget.

    We train with:
    - `custom_generate_function`: runs the interaction loop.
    - `custom_rm_function`: rewards correct answers with an early-turn bonus.
    - `loss_mask`: trains only on model-generated tokens, not environment feedback.
    """


@py_only
@markdown
def _run_instructions():
    """
    Run with:
    ```
    uv run python tutorials/rl/002_multiturn_number_guessing/002_multiturn_number_guessing.py
    ```
    """


@notebook_only
@shell("%uv pip install -q git+https://github.com/modal-projects/training-gym.git@main")
def _install():
    pass


@code
def _imports():
    import json
    import re

    from modal_training_gym import (
        DatasetConfig,
        DeploymentConfig,
        EvalConfig,
        EvalRowResult,
        ModelDeployment,
        Qwen3_4B,
        SlimeRecipe,
        TrainConfig,
        WandbConfig,
    )
    from modal_training_gym.common.checkpoint import list_checkpoints


@markdown
def _dataset_intro():
    """
    ## Build a deterministic guessing dataset

    Keep it simple:
    - train on odd targets
    - evaluate on even targets
    """


@code
def _dataset():
    _MAX_VALUE = 20
    _MAX_TURNS = 6
    _PROMPT = (
        "You are playing a number guessing game.\n"
        "The hidden integer is between 1 and 20.\n"
        "Return only guesses in this exact format: <answer>N</answer>\n"
        "After each guess, you will receive <feedback>higher</feedback> or "
        "<feedback>lower</feedback>, and must update your next guess accordingly.\n"
        "where N is an integer between 1 and 20."
    )

    TRAIN_TARGETS = list(range(1, _MAX_VALUE + 1, 2))
    TEST_TARGETS = list(range(2, _MAX_VALUE + 1, 2))

    class NumberGuessDataset(DatasetConfig):
        input_key = "messages"
        label_key = "label"
        apply_chat_template = True
        input_column = "prompt"

        def load(self):
            return [{"prompt": _PROMPT, "target": target} for target in TEST_TARGETS]

        def prepare(self, path: str, eval_paths: dict[str, str] | None = None):
            import os

            from datasets import Dataset

            os.makedirs(os.path.dirname(path), exist_ok=True)

            def _row(target: int) -> dict:
                return {
                    "messages": [{"role": "user", "content": _PROMPT}],
                    "label": json.dumps({"answer": target}),
                }

            train_rows = [_row(target) for target in TRAIN_TARGETS for _ in range(20)]
            eval_rows = [_row(target) for target in TEST_TARGETS]

            Dataset.from_list(train_rows).to_parquet(path)
            if eval_paths:
                for eval_path in eval_paths.values():
                    os.makedirs(os.path.dirname(eval_path), exist_ok=True)
                    Dataset.from_list(eval_rows).to_parquet(eval_path)

    train_dataset = NumberGuessDataset()
    eval_dataset = NumberGuessDataset()


@markdown
def _rollout_intro():
    """
    ## Multi-turn environment and reward

    `number_guess_generate` is the environment loop:
    - model generates `<answer>N</answer>`
    - environment appends `<feedback>higher|lower</feedback>` when incorrect
    - only model text is trained (`loss_mask=1`), feedback is masked out (`loss_mask=0`)

    Reward mirrors PipelineRL-style shaping:
    - success: `2.0 - 0.1 * (turns - 1)`
    - malformed output: `-2.0`
    - otherwise: `-1.0`
    """


@code
def _rollout_and_rm():
    _ANSWER_RE = re.compile(r"<answer>\s*(\d+)\s*</answer>", re.IGNORECASE)

    def _parse_label(sample) -> dict:
        raw = getattr(sample, "label", None)
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str):
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return {}
        return {}

    def _extract_answer(text: str) -> int | None:
        matches = list(_ANSWER_RE.finditer(text))
        if not matches:
            return None
        guess = int(matches[-1].group(1))
        if 1 <= guess <= _MAX_VALUE:
            return guess
        return None

    async def number_guess_generate(args, sample, sampling_params):
        from slime.rollout.sglang_rollout import GenerateState
        from slime.utils.http_utils import post
        from slime.utils.types import Sample

        label = _parse_label(sample)
        target = int(label.get("answer", 1))
        max_turns = int(getattr(args, "max_turns", _MAX_TURNS))

        state = GenerateState(args)
        url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

        prompt_ids = state.tokenizer(sample.prompt, add_special_tokens=False)["input_ids"]
        trajectory_text = ""
        response_segments: list[tuple[str, int]] = []

        success = False
        format_error = False
        turns_taken = max_turns
        final_status = Sample.Status.COMPLETED

        for turn in range(max_turns):
            output = await post(
                url,
                {
                    "text": f"{sample.prompt}\n{trajectory_text}".strip(),
                    "sampling_params": sampling_params,
                },
            )
            finish_type = output["meta_info"]["finish_reason"]["type"]
            if finish_type == "abort":
                sample.status = Sample.Status.ABORTED
                return sample

            model_text = output["text"]
            trajectory_text += model_text
            response_segments.append((model_text, 1))

            guess = _extract_answer(model_text)
            if guess is None:
                format_error = True
                turns_taken = turn + 1
                break

            if guess == target:
                success = True
                turns_taken = turn + 1
                break

            feedback = "higher" if guess < target else "lower"
            feedback_text = f"\n<feedback>{feedback}</feedback>\n"
            trajectory_text += feedback_text
            response_segments.append((feedback_text, 0))

            if finish_type == "length":
                final_status = Sample.Status.TRUNCATED
                break

        response_token_ids: list[int] = []
        loss_masks: list[int] = []
        for segment_text, trainable in response_segments:
            token_ids = state.tokenizer(
                segment_text,
                add_special_tokens=False,
            )["input_ids"]
            response_token_ids += token_ids
            loss_masks += [trainable] * len(token_ids)

        sample.tokens = prompt_ids + response_token_ids
        sample.response_length = len(response_token_ids)
        sample.response = trajectory_text
        sample.loss_mask = loss_masks
        sample.status = final_status

        sample_metadata = getattr(sample, "metadata", None)
        if not isinstance(sample_metadata, dict):
            sample_metadata = {}
        sample_metadata["guessing"] = {
            "target": target,
            "success": success,
            "format_error": format_error,
            "turns_taken": turns_taken,
        }
        sample.metadata = sample_metadata
        return sample

    def _trajectory_reward(success: bool, format_error: bool, turns_taken: int) -> float:
        if success:
            return float(2.0 - 0.1 * max(0, turns_taken - 1))
        if format_error:
            return -2.0
        return -1.0

    async def number_guess_rm(args, sample, **kwargs) -> float:
        sample_metadata = getattr(sample, "metadata", None)
        guessing_meta = sample_metadata.get("guessing", {}) if isinstance(sample_metadata, dict) else {}

        success = bool(guessing_meta.get("success", False))
        format_error = bool(guessing_meta.get("format_error", False))
        turns_taken = int(guessing_meta.get("turns_taken", getattr(args, "max_turns", _MAX_TURNS)))
        return _trajectory_reward(
            success=success,
            format_error=format_error,
            turns_taken=turns_taken,
        )


@markdown
def _eval_intro():
    """
    ## Offline multi-turn trajectory evaluator

    `EvalConfig` now supports `generic_eval_fn`, so we can plug in a full
    multi-turn evaluator per row while still using the standard eval runner.
    """


@code
def _eval_helpers():
    def run_guessing_trajectory(
        deployment: ModelDeployment,
        *,
        target: int,
        max_turns: int = _MAX_TURNS,
    ) -> dict:
        trace = ""
        for turn in range(max_turns):
            prompt = f"{_PROMPT}\n{trace}".strip()
            response = deployment.generate(
                prompt,
                chat_template_kwargs={"enable_thinking": False},
            )
            guess = _extract_answer(response)
            if guess is None:
                return {
                    "success": False,
                    "format_error": True,
                    "turns_taken": turn + 1,
                    "response": response,
                }
            if guess == target:
                return {
                    "success": True,
                    "format_error": False,
                    "turns_taken": turn + 1,
                    "response": response,
                }
            feedback = "higher" if guess < target else "lower"
            trace += f"{response}\n<feedback>{feedback}</feedback>\n"
        return {
            "success": False,
            "format_error": False,
            "turns_taken": max_turns,
            "response": trace,
        }

    def _resolve_target(example: dict) -> int:
        if "target" in example:
            return int(example["target"])

        raw = example.get("label")
        if isinstance(raw, dict):
            return int(raw.get("answer", 1))
        if isinstance(raw, str):
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                payload = {}
            return int(payload.get("answer", 1))
        return 1

    def guessing_generic_eval_fn(
        deployment: ModelDeployment,
        example: dict,
    ) -> EvalRowResult:
        target = _resolve_target(example)
        trajectory = run_guessing_trajectory(
            deployment,
            target=target,
            max_turns=_MAX_TURNS,
        )
        reward = _trajectory_reward(
            success=trajectory["success"],
            format_error=trajectory["format_error"],
            turns_taken=trajectory["turns_taken"],
        )
        return EvalRowResult(
            score=reward,
            response=trajectory["response"],
            metadata={
                "success": trajectory["success"],
                "format_error": trajectory["format_error"],
                "turns_taken": trajectory["turns_taken"],
                "target": target,
            },
        )

    def summarize_eval(eval_result) -> dict:
        rows = eval_result.rows
        success_rate = sum(1 for row in rows if row.metadata.get("success")) / max(
            len(rows), 1
        )
        mean_turns = sum(
            int(row.metadata.get("turns_taken", _MAX_TURNS)) for row in rows
        ) / max(len(rows), 1)
        return {
            "success_rate": float(success_rate),
            "mean_turns": float(mean_turns),
        }


@markdown
def _serve_base_intro():
    """
    ## Serve and evaluate the base model
    """


@code
def _serve_base():
    base_deployment = DeploymentConfig(model=Qwen3_4B()).serve()
    print(f"Base model URL: {base_deployment.url}")
    eval_config = EvalConfig(
        dataset=eval_dataset,
        generic_eval_fn=guessing_generic_eval_fn,
    )
    base_eval = eval_config.evaluate(base_deployment, debug=True)
    base_summary = summarize_eval(base_eval)
    print(f"Base success rate: {base_summary['success_rate']:.2%}")
    print(f"Base mean reward: {base_eval.mean:.3f}")
    print(f"Base mean turns:  {base_summary['mean_turns']:.2f}")


@markdown
def _train_intro():
    """
    ## Train with custom multi-turn rollout
    """


@code
def _train():
    training_run = TrainConfig(
        model=Qwen3_4B(),
        dataset=train_dataset,
        recipe=SlimeRecipe(
            wandb=WandbConfig(project="gym-tutorial", group="qwen3-4b-guessing-multiturn"),
            custom_generate_function=number_guess_generate,
            custom_rm_function=number_guess_rm,
            custom_config_path={
                "max_turns": _MAX_TURNS,
                "log_multi_turn": True,
            },
            num_rollout=20,
            rollout_batch_size=4,
            n_samples_per_prompt=2,
            rollout_max_response_len=96,
            global_batch_size=4,
            max_tokens_per_gpu=4096,
            save_interval=10,
            actor_num_nodes=1,
            actor_num_gpus_per_node=1,
            apply_chat_template_kwargs='{"enable_thinking": false}',
            image_overlay=lambda image: image.run_commands(
                "uv pip install --system datasets>=3.0.0",
            ),
        ),
    )
    print("Starting training...")
    train_result = training_run.train()
    print(f"Training run id: {train_result.training_run_id}")


@markdown
def _trained_eval_intro():
    """
    ## Evaluate trained checkpoint
    """


@code
def _trained_eval():
    checkpoint = list_checkpoints(train_result.training_run_id)[-1]
    trained_deployment = DeploymentConfig(
        model=Qwen3_4B(),
        checkpoint=checkpoint,
        app_name="qwen3-4b-guessing-multiturn-serve",
        served_model_name="qwen3-4b-guessing-multiturn",
    ).serve()
    print(f"Trained model URL: {trained_deployment.url}")

    trained_eval = eval_config.evaluate(trained_deployment, debug=True)
    trained_summary = summarize_eval(trained_eval)
    print(f"Trained success rate: {trained_summary['success_rate']:.2%}")
    print(f"Trained mean reward: {trained_eval.mean:.3f}")
    print(f"Trained mean turns:  {trained_summary['mean_turns']:.2f}")
    print(f"Base success rate:    {base_summary['success_rate']:.2%}")
    print(f"Base mean reward:     {base_eval.mean:.3f}")
    print(f"Base mean turns:      {base_summary['mean_turns']:.2f}")
