"""Tutorial source for `verl_qwen3_32b_gsm8k` — parsed by generate_tutorial.py.

Each `@markdown`-decorated function contributes a markdown cell (its
docstring); each `@code`-decorated function contributes a code cell (its
body, dedented). README content is pulled into markdown cells so it shows
up in both the .ipynb and (as `#` comments) in the .py.
"""

from tutorial_generator import code, markdown, notebook_only, py_only, shell


@markdown
def _title():
    """
    # verl: GRPO Fine-tuning Qwen3-32B on GSM8K

    Full-finetuning of Qwen3-32B via RLVR (GRPO) on GSM8K, using
    [verl](https://github.com/volcengine/verl) on Modal's multi-node
    infrastructure. 4 nodes × 8×H100, coordinated through a Ray cluster;
    verl itself is submitted via `JobSubmissionClient` and runs under
    Megatron + vLLM.
    """


@markdown
def _quickstart():
    """
    ## Quickstart

    Four stages, in order:

    1. `prep_dataset`        — runs verl's bundled GSM8K preprocessor.
    2. `download_model`      — snapshot_downloads Qwen3-32B HF weights.
    3. `convert_hf_to_mcore` — HF → Megatron-core via verl's converter.
    4. `train_multi_node`    — 4-node RLVR training; any positional args
       become Hydra overrides appended to the verl command (e.g.
       `-- trainer.total_epochs=20`).
    """


@notebook_only
@shell("%uv pip install -q git+https://github.com/modal-projects/training-gym.git@joy/initial-setup")

def _install():
    pass


@code
def _imports():
    from modal_training_gym.common.dataset import DatasetConfig
    from modal_training_gym.common.models import BaseModelType, Model
    from modal_training_gym.common.wandb import WandbConfig
    from modal_training_gym.frameworks.verl import (
        DATA_PATH,
        VerlConfig,
        VerlModalConfig,
        build_verl_app,
    )


@markdown
def _explain_dataset():
    """
    ## Define the dataset

    verl ships a GSM8K preprocessor at
    `verl/examples/data_preprocess/gsm8k.py`. Our `prepare()` shells out to
    it; the result is `train.parquet` + `test.parquet` under `DATA_PATH`,
    which the config's `data.train_files` / `data.val_files` point at.
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

            # verl is cloned to /root/verl during image build.
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

    Subclass `VerlConfig` and override class attributes. The launcher
    translates these into Hydra-style `key=value` overrides appended to
    `verl.trainer.main_ppo`. Containers (`dataset`, `model`, `wandb`) are
    read explicitly — verl's flag vocabulary (e.g.
    `actor_rollout_ref.actor.megatron.tensor_model_parallel_size`) doesn't
    line up with the common attr names 1:1.

    ### Parallelism (32 GPUs total = 4 nodes × 8 H100)

    | Pool    | Setting                                          |
    |---------|--------------------------------------------------|
    | Actor   | TP=8, PP=1, CP=1, EP=1 (Megatron)                |
    | Rollout | TP=4, DP=1, EP=1 (vLLM, async)                   |
    | Ref     | TP=8, PP=1 (Megatron, param offload on)          |

    ### Reward

    Binary GSM8K reward: 1.0 if the model's `#### N` answer matches ground
    truth exactly, else 0.0. Shipped with the framework at
    `modal_training_gym.frameworks.verl.gsm8k_reward:compute_reward`; the
    launcher resolves its on-disk path and hands it to verl's
    `custom_reward_function.path=` flag.
    """


@markdown
def _explain_reward():
    """
    ## Custom reward function (optional)

    verl loads a reward function from a `.py` file via
    `custom_reward_function.path=`. The default (binary GSM8K match) is
    shipped with the framework at
    `modal_training_gym.frameworks.verl.gsm8k_reward`. To override, set
    `reward_function_source` on your config to a Python source string, then
    run `app.upload_reward` before `app.train_multi_node`:

    - The source is written to a dedicated Modal volume as `custom_reward.py`.
    - `train_multi_node` mounts that volume and points verl at the file.
    - The file is a regular `.py` — importable from either a Python script
      or a Jupyter notebook for local testing.
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
    class _Verl(VerlConfig):
        # ── Containers ────────────────────────────────────────────────────────
        model = Model(BaseModelType.Qwen3_32B)
        dataset = GSM8KDataset(DATA_PATH)
        wandb = WandbConfig(
            project="verl_grpo_qwen3_32b",
            exp_name="qwen3_32b_megatron_gsm8k",
        )

        # ── Reward ────────────────────────────────────────────────────────────
        # Uploaded to the rewards volume by `app.upload_reward`. Leave empty
        # to fall back to the shipped GSM8K reward.
        reward_function_source = REWARD_SOURCE
        reward_function_name = "compute_reward"

        # ── Infrastructure ────────────────────────────────────────────────────
        n_nodes = 4
        gpus_per_node = 8

        # ── Data / algorithm ──────────────────────────────────────────────────
        adv_estimator = "grpo"
        max_prompt_length = 768
        max_response_length = 2048
        train_batch_size = 128

        # ── Actor ─────────────────────────────────────────────────────────────
        actor_strategy = "megatron"
        actor_lr = 1e-6
        ppo_mini_batch_size = 128
        actor_use_dynamic_bsz = True
        use_kl_loss = True
        kl_loss_coef = 0.001

        # Actor Megatron parallelism
        actor_tp_size = 8
        actor_pp_size = 1
        actor_cp_size = 1
        actor_ep_size = 1

        # ── Rollout (vLLM) ────────────────────────────────────────────────────
        n_rollouts_per_prompt = 4
        rollout_tp_size = 4
        rollout_gpu_memory_util = 0.25

        # ── Trainer ───────────────────────────────────────────────────────────
        test_freq = 5
        save_freq = -1
        resume_mode = "auto"


@markdown
def _build_section():
    """
    ## Build the Modal app
    """


@code
def _build_app():
    app = build_verl_app(
        modal=VerlModalConfig(gpu="H100"),
        verl=_Verl(),
    )


@markdown
def _run_section():
    """
    ## Run it
    """


@py_only
@markdown
def _run_cli():
    """
    From the CLI (in order):

    ```bash
    uv run modal run tutorials/verl_qwen3_32b_gsm8k/verl_qwen3_32b_gsm8k.py::app.prep_dataset
    uv run modal run tutorials/verl_qwen3_32b_gsm8k/verl_qwen3_32b_gsm8k.py::app.download_model
    uv run modal run tutorials/verl_qwen3_32b_gsm8k/verl_qwen3_32b_gsm8k.py::app.convert_hf_to_mcore

    # (Optional) upload custom reward source — re-run when you change it.
    uv run modal run tutorials/verl_qwen3_32b_gsm8k/verl_qwen3_32b_gsm8k.py::app.upload_reward

    uv run modal run --detach tutorials/verl_qwen3_32b_gsm8k/verl_qwen3_32b_gsm8k.py::app.train_multi_node

    # Pass Hydra overrides through after `--`:
    uv run modal run tutorials/verl_qwen3_32b_gsm8k/verl_qwen3_32b_gsm8k.py::app.train_multi_node -- trainer.total_epochs=20
    ```
    """


@notebook_only
@markdown
def _run_interactive():
    """
    Interactive — open an ephemeral app and call functions remotely:
    """


@notebook_only
@code
def _invoke():
    with app.run():
        # app.prep_dataset.remote()
        # app.download_model.remote()
        # app.convert_hf_to_mcore.remote()
        # app.upload_reward.remote()  # run once if you set reward_function_source
        app.train_multi_node.remote()
