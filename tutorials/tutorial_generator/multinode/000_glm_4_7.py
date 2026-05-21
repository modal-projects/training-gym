"""Tutorial source for `000_glm_4_7` — parsed by generate_tutorial.py."""

TUTORIAL_METADATA = {
    "framework": "`slime`",
    "cluster_shape": "8 x 8xH200",
    "summary": "GLM-4.7 355B MoE full-weight GSPO training on 64 GPUs with DAPO-Math-17k",
    "difficulty": "Advanced",
    "order": 30,
    "api_classes": [
        "GLM_4_7",
        "GLM_4_7_Recipe",
        "GLM_4_7_SglangRecipe",
        "DeploymentConfig",
        "TrainConfig",
        "TrainResult",
    ],
}


from tutorial_generator import code, markdown, notebook_only, py_only, shell


@markdown
def _intro():
    """
    # Multi-node GLM-4.7 full-weight training

    This tutorial runs full-weight GSPO (Group-Guided Self-Preference
    Optimization) on [GLM-4.7](https://huggingface.co/zai-org/GLM-4.7),
    a 355B-parameter Mixture-of-Experts model from Zhipu AI, using
    [slime](https://github.com/THUDM/slime) across **8 nodes (64 H200 GPUs)**.

    GLM-4.7 routes each token through 8 of 160 experts (~32B active
    parameters per token). Training at this scale requires:

    - **TP=8, PP=4, CP=2, EP=16** to shard the model across 64 GPUs
    - **EAGLE speculative decoding** on the sglang rollout engine
    - **CPU-offloaded optimizer** to fit the full optimizer states

    The `GLM_4_7_Recipe` preset wires all of this up — you attach a
    dataset and call `train()`.
    """


@py_only
@markdown
def _run_instructions():
    """
    To run the tutorial, run the following command:
    ```
    uv run python tutorials/multinode/000_glm_4_7/000_glm_4_7.py
    ```
    """


@notebook_only
@shell("%uv pip install -q git+https://github.com/modal-projects/training-gym.git@main")
def _install():
    pass


@code
def _imports():
    from modal_training_gym import (
        DeploymentConfig,
        GLM_4_7,
        HuggingFaceDataset,
        TrainConfig,
        list_checkpoints,
    )
    from modal_training_gym.train_recipes.slime_recipe import GLM_4_7_Recipe
    from modal_training_gym.deploy_recipes.sglang_recipe import GLM_4_7_SglangRecipe


@markdown
def _dataset_intro():
    """
    ## Dataset

    We use [DAPO-Math-17k](https://huggingface.co/datasets/zhuzilin/dapo-math-17k),
    a collection of math competition problems with verifiable answers.
    The `deepscaler` reward model built into slime checks whether the
    model's response matches the reference answer.
    """


@code
def _define_dataset():
    class DAPOMath(HuggingFaceDataset):
        hf_repo = "zhuzilin/dapo-math-17k"
        input_key = "prompt"
        label_key = "label"
        output_format = "jsonl"
        apply_chat_template = True
        always_prepare = True

    dataset = DAPOMath()


@markdown
def _model_intro():
    """
    ## Model and recipe

    `GLM_4_7()` points at `zai-org/GLM-4.7` on HuggingFace and carries
    the full architecture spec (92 layers, 5120 hidden, 96 attention
    heads, 160 MoE experts).

    `GLM_4_7_Recipe()` is the training preset: 8 colocated H200 nodes,
    GSPO algorithm, EAGLE speculative rollout, CPU-offloaded Adam, and
    DeepEP-accelerated expert dispatch. Override any field to customize.
    """


@code
def _define_model():
    model = GLM_4_7()

    recipe = GLM_4_7_Recipe(
        rm_type="deepscaler",
    )

    print(f"Model: {model.model_name}")
    print(f"Nodes: {recipe.actor_num_nodes}, GPUs/node: {recipe.actor_num_gpus_per_node}")
    print(f"Parallelism: TP={recipe.tensor_model_parallel_size}, "
          f"PP={recipe.pipeline_model_parallel_size}, "
          f"CP={recipe.context_parallel_size}, "
          f"EP={recipe.expert_model_parallel_size}")
    print(f"Algorithm: {recipe.advantage_estimator}")


@markdown
def _train_intro():
    """
    ## Train

    `TrainConfig.train()` builds the Modal app, downloads the model
    weights, prepares the dataset, and launches multi-node training.
    The first run downloads ~710 GB of model weights into the shared
    HuggingFace cache volume — subsequent runs reuse the cache.
    """


@code
def _run_training():
    training_run = TrainConfig(
        model=model,
        dataset=dataset,
        recipe=recipe,
    )

    print(f"Training run: {training_run.training_run_id}")
    print(f"Total nodes: {recipe.total_nodes}")
    print("--- Starting training... ---")
    train_result = training_run.train()
    print("--- Training complete ---")


@markdown
def _serve_intro():
    """
    ## Serve the trained checkpoint

    After training, serve the checkpoint with SGLang for inference.
    `GLM_4_7_SglangRecipe` defaults to 8xH200 with TP=8 — enough
    to hold the full 355B model in BF16.
    """


@code
def _serve_checkpoint():
    checkpoint = list_checkpoints(train_result.training_run_id)[-1]
    print(f"Checkpoint: {checkpoint.path}")

    deployment = DeploymentConfig(
        model=GLM_4_7(),
        checkpoint=checkpoint,
        recipe=GLM_4_7_SglangRecipe(),
        app_name="glm-4-7-serve",
        served_model_name="glm-4-7",
    ).serve()
    print(f"Deployed to {deployment.url}")


@notebook_only
@markdown
def _try_it():
    """
    Let's test the trained model with a math problem.
    """


@notebook_only
@code
def _try_generation():
    response = deployment.generate(
        "Let $p$ be a prime number. Find the number of integers $n$ "
        "with $1 \\le n \\le p^2$ such that $n^{p-1} \\equiv 1 \\pmod{p^2}$.",
    )
    print(response)
