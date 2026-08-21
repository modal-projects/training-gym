"""Tutorial source for `002_glm_4_7` — parsed by generate_tutorial.py."""

TUTORIAL_METADATA = {
    "framework": "`slime`",
    "cluster_shape": "8 x 8xH200",
    "summary": "Multi-node full-weight training",
    "difficulty": "Advanced",
    "order": 30,
    "api_classes": [
        "Endpoint",
        "GLM_4_7_Recipe",
        "GLM_4_7",
        "HuggingFaceDataset",
        "TrainConfig",
    ],
}


from tutorial_generator import code, markdown, notebook_only, py_only, shell


@markdown
def _intro():
    """
    # Frontier-scale training

    When you're decided that you need to train a frontier-scale model
    for your workload, you're probably looking for 1) the beefiest
    hardware you can get, and 2) guarantees that your runs won't crash.
    This tutorial trains [GLM-4.7](https://huggingface.co/zai-org/GLM-4.7)
    across 8 nodes with 8 H200s each for a total of 64 GPUs using full-weight
    [Group Sequence Policy Optimization](https://arxiv.org/abs/2507.18071) (GSPO)
    on the
    [zhuzilin/dapo-math-17k](https://huggingface.co/datasets/zhuzilin/dapo-math-17k)
    Huggingface dataset. And what do you know: it's easy as pie (possibly even easier).
    """


@py_only
@markdown
def _run_instructions():
    """
    Run with:

    ```
    uv run tutorials/multinode/002_glm_4_7/002_glm_4_7.py
    ```
    """


@notebook_only
@shell(
    "import importlib.util\n"
    "\n"
    "if importlib.util.find_spec('modal_training_gym') is None:\n"
    "    %uv pip install -q git+https://github.com/modal-projects/training-gym.git@main"
)
def _install():
    pass


@code
def _imports():
    from modal_training_gym import (
        Endpoint,
        GLM_4_7,
        HuggingFaceDataset,
        TrainConfig,
        list_checkpoints,
    )
    from modal_training_gym.train_recipes.slime_recipe import GLM_4_7_Recipe


@markdown
def _setup_intro():
    """
    ## Set up

    We'll quickly set up the model, dataset, and training recipe. All that you successfully
    train this model is encapsulated in the model and recipe classes. However, we surface
    some important parameters to make the run correct and fast.
    """

@code
def _setup():
    model = GLM_4_7()

    class MathDataset(HuggingFaceDataset):
        hf_repo = "zhuzilin/dapo-math-17k"
        input_key = "prompt"
        label_key = "label"
        output_format = "jsonl"
        apply_chat_template = True
        always_prepare = True

    train_dataset = MathDataset(hf_split="train[:2000]")
    recipe = GLM_4_7_Recipe(
        rm_type="deepscaler",
    )

    print(f"training and rollout gpus colocated: {recipe.colocate}")
    print(f"training nodes: {recipe.actor_num_nodes}, gpus/node: {recipe.actor_num_gpus_per_node}")
    print(f"rollout gpus: {recipe.rollout_num_gpus}, rollout gpus/engine: {recipe.rollout_num_gpus_per_engine}")
    print(f"parallelism: tp={recipe.tensor_model_parallel_size}, pp={recipe.pipeline_model_parallel_size}, "
          f"cp={recipe.context_parallel_size}, ep={recipe.expert_model_parallel_size}")
    print(f"optimizer cpu offload: {recipe.optimizer_cpu_offload}")


@markdown
def _train_intro():
    """
    ## Kick off training

    Let's begin, shall we?
    """


@code
def _train():
    config = TrainConfig(
        model=model,
        dataset=train_dataset,
        recipe=recipe,
    )

    run = config.launch()
    print(f"run id: {run.training_run_id}")


@markdown
def _test_intro():
    """
    ## Test out the trained model

    We'll spin up an [Endpoint](https://modal.com/docs/guide/endpoints)
    and see how the trained model does.
    """


@code
def _serve_checkpoint():
    result = run.result()
    checkpoint = list_checkpoints(result.training_run_id)[-1]
    print(f"checkpoint: {checkpoint.path}")

    trained_deployment = Endpoint.launch(
        model, checkpoint, unauthenticated=True, recreate_if_existing=True
    )
    trained_deployment.wait_until_ready(timeout=45 * 60)
    print(f"checkpoint deployed to {trained_deployment.url}")

    msg = trained_deployment.chat(
        [
            {
                "role": "user",
                "content": (
                    "Let $p$ be a prime number. Find the number of integers $n$ "
                    "with $1 \\le n \\le p^2$ such that $n^{p-1} \\equiv 1 \\pmod{p^2}$."
                ),
            }
        ],
    )
    print(msg.get("content") or msg.get("reasoning_content") or "")
