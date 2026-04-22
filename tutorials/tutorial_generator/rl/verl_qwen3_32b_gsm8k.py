"""Tutorial source for `verl_qwen3_32b_gsm8k` — parsed by generate_tutorial.py.

Each `@markdown`-decorated function contributes a markdown cell (its
docstring); each `@code`-decorated function contributes a code cell (its
body, dedented). README content is pulled into markdown cells so it shows
up in both the .ipynb and (as `#` comments) in the .py.
"""

TUTORIAL_METADATA = {
    'framework': '`verl`',
    'cluster_shape': '4 × 8×H100',
    'summary': 'Qwen3-32B GRPO on GSM8K (Megatron + vLLM)',
    'difficulty': 'Advanced',
    'order': 40,
}


from tutorial_generator import code, markdown, notebook_only, py_only, shell


@markdown
def _intro():
    """
    # Qwen3-32B GRPO on GSM8K with verl on Modal

    **What verl is.** [verl](https://github.com/volcengine/verl) is an RL
    post-training framework from ByteDance/Volcengine. It pairs
    **Megatron** (training) with **vLLM** (rollouts) and submits jobs to a
    **Ray** cluster via `JobSubmissionClient`. Config is Hydra-style;
    every knob is a `key=value` override on `verl.trainer.main_ppo`.

    **verl vs slime.** Same family (GRPO post-training, Ray-coordinated,
    Megatron-trained), different rollout engine (vLLM instead of SGLang)
    and different config vocabulary (Hydra overrides instead of a flat CLI
    flag namespace). verl tends to be the choice when the deployment story
    also wants vLLM; slime is the choice when SGLang or an async reward
    loop matters.

    **What this tutorial does.** Full-finetunes Qwen3-32B via GRPO on
    GSM8K on 4 nodes × 8×H100 (32 GPUs). Demonstrates verl's
    reward-function override hook (`custom_reward_function.path=`) by
    shipping a partial-credit GSM8K reward from user-supplied source.
    For the shared primitives (`DatasetConfig`, `Model`, 3-stage pipeline)
    see [`quickstart`](../../intro/quickstart/quickstart.ipynb).

    **What you'll need.**
    - Access to Modal's multi-node training preview (4 × 8×H100).
    - `wandb` Modal secret.
    - Patience — 32B full-finetune is a multi-hour run.

    **What to watch.** W&B project `verl_grpo_qwen3_32b`, experiment
    `qwen3_32b_megatron_gsm8k`. Look for rollout reward climbing; the
    custom reward here is partial-credit so expect early scores in the
    0.1–0.3 range before exact matches start accumulating.
    """


@markdown
def _pipeline():
    """
    ## Pipeline

    Five stages (one more than the base three — verl needs a separate
    HF → Megatron-core conversion step, and reward source is uploaded
    once through a dedicated volume):

    1. `prep_dataset` — shells out to verl's bundled GSM8K preprocessor.
    2. `download_model` — snapshots Qwen3-32B into the HF cache volume.
    3. `convert_hf_to_mcore` — runs verl's HF → Megatron-core converter.
    4. `upload_reward` (optional) — writes the user-supplied reward
       source to a Modal volume that `train_multi_node` mounts.
    5. `train_multi_node` — 4-node GRPO; positional args after `--`
       become Hydra overrides (e.g. `-- trainer.total_epochs=20`).
    """


@notebook_only
@shell("%uv pip install -q git+https://github.com/modal-projects/training-gym.git@joy/initial-setup")

def _install():
    pass


@code
def _imports():
    import modal

    from modal_training_gym.common.dataset import DatasetConfig
    from modal_training_gym.common.models import Qwen3_32B
    from modal_training_gym.common.wandb import WandbConfig
    from modal_training_gym.frameworks.verl import (
        DATA_PATH,
        VerlConfig,
        VerlFrameworkConfig,
    )


@markdown
def _explain_dataset():
    """
    ## Define the dataset

    verl ships a GSM8K preprocessor at
    `verl/examples/data_preprocess/gsm8k.py` that produces the exact
    parquet schema its trainer expects. Our `prepare()` just shells out
    to it — no in-Python preprocessing — and the resulting
    `train.parquet` / `test.parquet` land at `DATA_PATH`, where the
    config's `data.train_files` / `data.val_files` point at them.

    (verl is cloned to `/root/verl` during image build, so the path is
    stable inside the container.)
    """


@code
def _define_dataset():
    class GSM8KDataset(DatasetConfig):
        """verl-flavored GSM8K: parquet files built by verl's own script."""

        def __init__(self, data_path):
            self._data_path = str(data_path)
            self.prompt_data = f"{self._data_path}/train.parquet"
            self.eval_prompt_data = f"{self._data_path}/test.parquet"

        def prepare(self):
            import subprocess

            subprocess.run(
                [
                    "python",
                    "/root/verl/examples/data_preprocess/gsm8k.py",
                    "--local_dir",
                    self._data_path,
                ],
                check=True,
            )


@markdown
def _explain_config():
    """
    ## Define the experiment

    `VerlFrameworkConfig` holds verl-specific knobs; the launcher
    translates them into Hydra overrides appended to
    `verl.trainer.main_ppo`. verl's flag namespace doesn't line up 1:1
    with the common attr names (it has separate actor / rollout / ref
    pools, each with its own parallelism), so the common containers
    (`dataset`, `model`, `wandb`) are read explicitly rather than
    auto-forwarded.

    ### Parallelism (32 GPUs total = 4 nodes × 8 H100)

    verl splits GPUs into three logical pools:

    | Pool    | Setting                                   | Notes |
    |---------|-------------------------------------------|-------|
    | Actor   | TP=8, PP=1, CP=1, EP=1 (Megatron)         | Trains Qwen3-32B under tensor parallelism |
    | Rollout | TP=4, DP=1, EP=1 (vLLM, async)            | Serves the current actor weights for rollouts |
    | Ref     | TP=8, PP=1 (Megatron, param offload on)   | Frozen reference model for KL |

    `rollout_gpu_memory_util=0.25` keeps vLLM's KV-cache small so actor
    and rollout can coexist on the same devices.

    ### RL objective

    - `adv_estimator="grpo"` — same GRPO as the slime tutorials.
    - `use_kl_loss=True`, `kl_loss_coef=0.001` — light KL leash against
      the reference model.
    - `actor_lr=1e-6` — small LR for a 32B full-finetune.

    ### Reward

    Default: verl's built-in binary GSM8K match (1.0 for exact, 0.0
    otherwise), shipped at
    `modal_training_gym.frameworks.verl.gsm8k_reward:compute_reward`.
    This tutorial overrides it with partial-credit source (see below).
    """


@markdown
def _explain_reward():
    """
    ## Custom reward (optional)

    verl loads a reward function from a `.py` file via
    `custom_reward_function.path=`. To override the default, set
    `reward_function_source` on your `VerlFrameworkConfig` to a Python
    source string, then run `app.upload_reward` before
    `app.train_multi_node`. The launcher:

    1. Writes the source to a dedicated Modal volume as
       `custom_reward.py`.
    2. Mounts that volume on the training container.
    3. Points verl at the file via the `custom_reward_function.path=`
       override.

    The file is a plain `.py` — importable for local testing from either
    a Python script or a Jupyter notebook before you ship it.

    The snippet below gives partial credit (0.1) for any numeric answer
    and full credit (1.0) for exact matches — useful early in training
    when the model rarely gets the full answer right but is learning to
    output *some* number.
    """


@code
def _reward_source():
    # Dedent + strip the leading newline so the file on disk has clean indentation.
    REWARD_SOURCE = '''
    def compute_reward(data_source, solution_str, ground_truth, extra_info, **kwargs):
        """Partial-credit GSM8K reward: 1.0 for exact match, 0.1 for any number match."""
        import re

        nums = re.findall(r"-?\\d+(?:\\.\\d+)?", solution_str)
        if not nums:
            return 0.0
        final = nums[-1].rstrip(".").replace(",", "")
        try:
            return 1.0 if float(final) == float(ground_truth) else 0.1
        except ValueError:
            return 0.0
    '''
    import textwrap

    REWARD_SOURCE = textwrap.dedent(REWARD_SOURCE).lstrip("\n")


@code
def _define_config():
    verl_framework_config = VerlFrameworkConfig(
        gpu="H100",
        reward_function_source=REWARD_SOURCE,
        reward_function_name="compute_reward",
        n_nodes=4,
        gpus_per_node=8,
        adv_estimator="grpo",
        max_prompt_length=768,
        max_response_length=2048,
        train_batch_size=16,
        actor_strategy="megatron",
        actor_lr=1e-6,
        ppo_mini_batch_size=16,
        actor_use_dynamic_bsz=True,
        use_kl_loss=True,
        kl_loss_coef=0.001,
        actor_tp_size=8,
        actor_pp_size=1,
        actor_cp_size=1,
        actor_ep_size=1,
        n_rollouts_per_prompt=1,
        rollout_tp_size=4,
        rollout_gpu_memory_util=0.25,
        test_freq=-1,
        save_freq=-1,
        resume_mode="auto",
    )

    my_training_run = VerlConfig(
        dataset=GSM8KDataset(DATA_PATH),
        model=Qwen3_32B(),
        wandb=WandbConfig(
            project="verl_grpo_qwen3_32b",
            exp_name="qwen3_32b_megatron_gsm8k",
        ),
        framework_config=verl_framework_config,
    )


@markdown
def _build_section():
    """
    ## Build and run

    `build_app()` returns a Modal app with the five functions named
    above. See [`quickstart`](../../intro/quickstart/quickstart.ipynb) for the
    general pattern.
    """


@code
def _build_app():
    app = my_training_run.build_app()


@py_only
@markdown
def _run_cli():
    """
    From the CLI (in order):

    ```bash
    uv run modal run tutorials/rl/verl_qwen3_32b_gsm8k/verl_qwen3_32b_gsm8k.py::app.prep_dataset
    uv run modal run tutorials/rl/verl_qwen3_32b_gsm8k/verl_qwen3_32b_gsm8k.py::app.download_model
    uv run modal run tutorials/rl/verl_qwen3_32b_gsm8k/verl_qwen3_32b_gsm8k.py::app.convert_hf_to_mcore

    # (Optional) upload custom reward source — re-run when you change it.
    uv run modal run tutorials/rl/verl_qwen3_32b_gsm8k/verl_qwen3_32b_gsm8k.py::app.upload_reward

    uv run modal run --detach tutorials/rl/verl_qwen3_32b_gsm8k/verl_qwen3_32b_gsm8k.py::app.train_multi_node

    # Pass Hydra overrides through after `--`:
    uv run modal run tutorials/rl/verl_qwen3_32b_gsm8k/verl_qwen3_32b_gsm8k.py::app.train_multi_node -- trainer.total_epochs=20
    ```
    """


@notebook_only
@markdown
def _run_interactive():
    """
    Interactive — one stage per cell:
    """


@notebook_only
@code
def _invoke_prep_dataset():
    with app.run():
        app.prep_dataset.remote()


@notebook_only
@code
def _invoke_download_model():
    with app.run():
        app.download_model.remote()


@notebook_only
@code
def _invoke_convert_hf_to_mcore():
    with app.run():
        app.convert_hf_to_mcore.remote()


@notebook_only
@code
def _invoke_upload_reward():
    with app.run():
        app.upload_reward.remote()


@notebook_only
@code
def _invoke_train_multi_node():
    with app.run():
        app.train_multi_node.remote()
