"""Tutorial source for `slime_haiku` — parsed by generate_tutorial.py.

Qwen3-4B GRPO on haiku poems. Demonstrates a custom async reward model
that combines deterministic structure scoring (5-7-5 syllable count) with
an optional LLM-as-judge style score. The judge is built on
`modal_training_gym.common.llm_judge.LlmJudge`; the full reward function
lives in the tutorial-local `haiku` module (sibling file). The slime
launcher ships that module into the training image via
`local_python_sources=["haiku"]`, so SLIME can resolve
`custom_rm_path="haiku.haiku_rm"` at training time.
"""

TUTORIAL_METADATA = {
    'framework': '`slime`',
    'cluster_shape': '1 × 8×H200',
    'summary': 'Qwen3-4B GRPO on haiku poems — structure score + LLM judge',
    'order': 15,
}


from tutorial_generator import code, markdown, notebook_only, py_only, shell


@markdown
def _intro():
    """
    # Qwen3-4B GRPO on Haiku with SLIME on Modal

    Trains Qwen3-4B with GRPO to write haiku poems about Modal-flavored
    topics, on 1 node × 8×H200 (actor and rollout colocated). The reward
    model has two parts:

    1. **Structure** — deterministic 5-7-5 syllable count via CMUdict.
    2. **Style** — optional `LlmJudge`-based score. Reads the judge URL
       from `LLM_JUDGE_URL`; when unset, the reward falls back to
       structure only so this tutorial is runnable without a separate
       judge deployment.

    Both halves live in `modal_training_gym.common.haiku` and plug into
    SLIME via `custom_rm_path`. Invoke any function on the returned `app`
    via `modal run`, or interactively with `app.run()` + `.remote()`.
    """


@notebook_only
@shell("%uv pip install -q git+https://github.com/modal-projects/training-gym.git@joy/initial-setup")
def _install():
    pass


@code
def _imports():
    import modal

    import haiku  # sibling module — defines the custom reward function

    from modal_training_gym.common.dataset import DatasetConfig
    from modal_training_gym.common.models import Qwen3_4B
    from modal_training_gym.common.wandb import WandbConfig
    from modal_training_gym.common.serve_vllm import build_vllm_serve_app
    from modal_training_gym.frameworks.slime import ModalConfig, SlimeConfig
    from modal_training_gym.frameworks.slime.config import DATA_PATH


@markdown
def _explain_dataset():
    """
    ## Define the dataset

    Subclass `DatasetConfig`, point it at the shared data volume, and
    implement `prepare()` to materialize the
    [`statworx/haiku`](https://huggingface.co/datasets/statworx/haiku)
    dataset as parquet. The system prompt nudges the model toward the
    Modal vocabulary list so finished poems are both structurally valid
    haiku and thematically on-topic.
    """


@code
def _define_dataset():
    class HaikuDataset(DatasetConfig):
        input_key = "messages"
        label_key = "label"
        apply_chat_template = True
        rollout_shuffle = True
        rm_type = "async_rm"

        def __init__(self, data_path, hf_checkpoint):
            self._data_path = str(data_path)
            self._hf_checkpoint = hf_checkpoint
            self.prompt_data = f"{self._data_path}/haiku/train.parquet"
            self.eval_prompt_data = ["haiku", f"{self._data_path}/haiku/test.parquet"]

        def prepare(self):
            import os

            from datasets import load_dataset
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(self._hf_checkpoint)
            ds = load_dataset("statworx/haiku")

            vocab_str = ", ".join(haiku.MODAL_VOCABS)
            system_prompt = (
                "You are a haiku poet. You will be given a prompt and you "
                "will need to write a haiku about the prompt. Try to "
                "incorporate these words into the haiku if possible: "
                f"{vocab_str}"
            )

            def to_chat(example):
                keyword = example["keywords"].lower()
                question = f"Write me a haiku about {keyword}."
                messages = [
                    {"content": system_prompt, "role": "system"},
                    {"content": question, "role": "user"},
                ]
                return {
                    "question": question,
                    "label": example["text"],
                    "messages": messages,
                    "prompt": tokenizer.apply_chat_template(
                        messages, tokenize=False, enable_thinking=False
                    ),
                }

            train_ds = ds["train"]
            test_size = min(1000, int(len(train_ds) * 0.2))
            test_ds = train_ds.select(range(len(train_ds) - test_size, len(train_ds)))
            train_ds = train_ds.select(range(len(train_ds) - test_size))

            os.makedirs(f"{self._data_path}/haiku", exist_ok=True)
            train_ds.map(to_chat, remove_columns=["keywords"]).to_parquet(
                f"{self._data_path}/haiku/train.parquet"
            )
            test_ds.map(to_chat, remove_columns=["keywords"]).to_parquet(
                f"{self._data_path}/haiku/test.parquet"
            )


@markdown
def _explain_reward():
    """
    ## Reward model

    The sibling `haiku.py` module defines `haiku_rm`, an async reward
    function matching the signature SLIME expects:
    `(args, sample, **kwargs) -> float`. It composes:

    - `score_haiku_structure(...)` — pure-Python 5-7-5 check via CMUdict.
    - `HaikuStyleJudge(LlmJudge)` — a thin rubric subclass of the shared
      `LlmJudge`; posts a prompt to an OpenAI-compatible chat endpoint and
      parses the numeric score out of the reply.

    To wire up the style score, stand up a vLLM server somewhere reachable
    from Modal and point the training container at it via `LLM_JUDGE_URL`.
    The structure score alone is enough to get a coherent run going, which
    is what the tutorial config below does.
    """


@markdown
def _explain_config():
    """
    ## Define the experiment

    `rm_type="async_rm"` + `custom_rm_path="haiku.haiku_rm"` tells SLIME
    to load the async reward function by Python import path. Declaring
    `local_python_sources=["haiku"]` on `ModalConfig` tells the slime
    launcher to package the sibling `haiku.py` into the training image —
    after that, the module resolves under its plain top-level name in the
    remote container. The `image_run_commands` entry installs `aiohttp`
    and `nltk` (needed by the reward model's HTTP client and syllable
    counter) into the SLIME base image.
    """


@code
def _define_config():
    base_model = Qwen3_4B()
    my_training_run = SlimeConfig(
        model=base_model,
        dataset=HaikuDataset(DATA_PATH, base_model.model_name),
        wandb=WandbConfig(project="slime-grpo", group="qwen3-4b-haiku"),
        ref_load=base_model.model_name,
        megatron_to_hf_mode="bridge",
        actor_num_nodes=1,
        actor_num_gpus_per_node=8,
        colocate=True,
        rm_type="async_rm",
        custom_rm_path="haiku.haiku_rm",
        num_rollout=50,
        rollout_batch_size=128,
        rollout_max_response_len=300,
        rollout_temperature=1.0,
        rollout_skip_special_tokens=True,
        rollout_num_gpus_per_engine=2,
        sglang_mem_fraction_static=0.7,
        n_samples_per_prompt=8,
        global_batch_size=64,
        apply_chat_template_kwargs='{"enable_thinking": false}',
        eval_interval=20,
        n_samples_per_eval_prompt=8,
        eval_max_response_len=300,
        eval_top_p=1.0,
        tensor_model_parallel_size=2,
        sequence_parallel=True,
        use_dynamic_batch_size=True,
        max_tokens_per_gpu=9216,
        recompute_granularity="full",
        recompute_method="uniform",
        recompute_num_layers=1,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        accumulate_allreduce_grads_in_fp32=True,
        attention_softmax_in_fp32=True,
        optimizer="adam",
        lr=1e-6,
        lr_decay_style="constant",
        weight_decay=0.1,
        adam_beta1=0.9,
        adam_beta2=0.98,
        advantage_estimator="grpo",
        eps_clip=0.2,
        eps_clip_high=0.28,
        use_kl_loss=True,
        kl_loss_coef=0.0,
        kl_loss_type="low_var_kl",
        entropy_coef=0.0,
        save="/checkpoints/qwen3-4b-haiku",
        save_interval=10,
        modal=ModalConfig(
            gpu="H200",
            local_python_sources=["haiku"],
            image_run_commands=[
                "uv pip install --system aiohttp nltk>=3.8.0",
                "python -c \"import nltk; nltk.download('cmudict', quiet=True)\"",
            ],
        ),
    )


@markdown
def _build_section():
    """
    ## Build the Modal app
    """


@code
def _build_app():
    app = my_training_run.build_app()


@markdown
def _run_section():
    """
    ## Run it
    """


@py_only
@markdown
def _run_cli():
    """
    Training (see the **Serve** section below for hosting the finished
    checkpoint):

    ```bash
    uv run modal run tutorials/slime_haiku/slime_haiku.py::app.download_model
    uv run modal run tutorials/slime_haiku/slime_haiku.py::app.prepare_dataset
    uv run modal run --detach tutorials/slime_haiku/slime_haiku.py::app.train
    ```

    Set `LLM_JUDGE_URL` (and optionally `LLM_JUDGE_MODEL`) on the train
    function's environment to enable the style score; omit it for a
    structure-only run.
    """


@notebook_only
@markdown
def _run_interactive():
    """
    Interactive — open an ephemeral app and run one stage per cell:
    """


@notebook_only
@code
def _invoke_download_model():
    with modal.enable_output():
        with app.run():
            app.download_model.remote()


@notebook_only
@code
def _invoke_prepare_dataset():
    with modal.enable_output():
        with app.run():
            app.prepare_dataset.remote()


@notebook_only
@code
def _invoke_train():
    with modal.enable_output():
        with app.run():
            app.train.remote()


@markdown
def _serve_section():
    """
    ## Serve the trained model

    Once training finishes, the `slime-checkpoints` volume holds the saved
    weights under `/checkpoints/qwen3-4b-haiku/`. SLIME bridge mode writes
    HF-format checkpoints per save interval as `iter_NNNNNNN_hf/`; point
    `build_vllm_serve_app` at whichever iteration you want to host. The
    helper returns a second Modal app whose `VllmServer` class starts vLLM
    against the checkpoint and exposes it through a Flash endpoint.
    """


@code
def _build_serve_app():
    serve_app = build_vllm_serve_app(
        app_name="serve-haiku-model",
        model_path="/checkpoints/qwen3-4b-haiku/iter_0000049_hf",
        served_model_name="qwen3-4b-haiku",
        gpu="H100",
        n_gpu=1,
        flash_region="us-east",
        extra_vllm_args=["--enforce-eager", "--reasoning-parser", "qwen3"],
    )


@py_only
@markdown
def _serve_cli():
    """
    Deploy the serving app (separate from the training app):

    ```bash
    uv run modal deploy tutorials/slime_haiku/slime_haiku.py::serve_app
    ```

    Query the Flash endpoint with any OpenAI-compatible client — the URL
    shape is
    `https://<workspace>--serve-haiku-model-vllmserver.us-east.modal.direct`:

    ```bash
    curl https://<workspace>--serve-haiku-model-vllmserver.us-east.modal.direct/v1/chat/completions \\
      -H "Content-Type: application/json" \\
      -d '{
            "model": "qwen3-4b-haiku",
            "messages": [{"role": "user", "content": "Write me a haiku about cat."}],
            "temperature": 0.0
          }'
    ```
    """


@notebook_only
@markdown
def _serve_interactive():
    """
    Deploy the serving app from a notebook — `modal.enable_output()` +
    `serve_app.deploy()` does the same thing as `modal deploy`:
    """


@notebook_only
@code
def _invoke_deploy_serve():
    with modal.enable_output():
        serve_app.deploy()
