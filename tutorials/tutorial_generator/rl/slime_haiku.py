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
    'difficulty': 'Intermediate',
    'order': 15,
}


from tutorial_generator import code, markdown, notebook_only, py_only, shell


@markdown
def _intro():
    """
    # Qwen3-4B GRPO on Haiku with SLIME on Modal

    This tutorial teaches [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B)
    to write 5-7-5 haiku poems about Modal-flavored topics. The training
    algorithm is **GRPO** (Group Relative Policy Optimization, the method
    popularized by DeepSeek-R1): for each prompt, the model generates a
    group of candidate responses, each response is scored by a reward
    model, and the policy is updated to prefer responses that scored
    above the group's mean.

    The interesting piece is the reward model, which has two halves:

    1. **Structure** — deterministic, cheap, and always on. A 5-7-5
       syllable check via CMUdict. No network, no GPU, runs inside the
       training loop.
    2. **Style** — an LLM-as-judge score. A separately-deployed vLLM
       server rates the poem on relevance, poetic quality, and Modal
       vocabulary usage. Optional: activated only when `LLM_JUDGE_URL`
       is set in the training environment, so the tutorial is runnable
       end-to-end without standing up a second service.

    Both pieces are wired into SLIME through its `custom_rm_path` hook,
    which points at an async function in the sibling `haiku.py` module.

    The workflow has four stages:

    | Stage | What it does | Where |
    |---|---|---|
    | `download_model` | Pulls `Qwen/Qwen3-4B` into the HF cache volume | 1×H200 |
    | `prepare_dataset` | Downloads `statworx/haiku` and writes train/test parquet | CPU |
    | `train` | GRPO training loop over the haiku prompts | 1×8×H200, colocated |
    | `serve_app` (separate) | Hosts the finished checkpoint via vLLM + Flash | 1×H100 |

    Invoke any function on the returned training `app` via `modal run`
    from the CLI, or interactively with `app.run()` + `.remote()` from a
    notebook.
    """


@notebook_only
@shell("%uv pip install -q git+https://github.com/modal-projects/training-gym.git@joy/initial-setup")
def _install():
    pass


@markdown
def _explain_imports():
    """
    ## Imports

    Three import groups:

    - **`modal`** — needed only for the interactive cells below
      (`modal.enable_output()`, `app.run()`). The training app itself is
      built by the launcher, so the tutorial body never touches the
      Modal SDK directly.
    - **`haiku`** — sibling file (`tutorials/rl/slime_haiku/haiku.py`).
      Defines the custom reward function SLIME calls, plus the
      `MODAL_VOCABS` list the dataset prep borrows for its system
      prompt. Lives next to this file (instead of in
      `modal_training_gym.common`) because the rubric is tutorial-
      specific; the slime launcher ships it into the training image
      automatically via `ModalConfig.local_python_sources=["haiku"]`.
    - **`modal_training_gym.*`** — shared containers (`DatasetConfig`,
      `Qwen3_4B`, `WandbConfig`), the vLLM serving helper, and SLIME's
      framework-specific config + launcher classes. `DATA_PATH` is the
      canonical mount point (`/data`) for the shared data volume on
      every SLIME container.
    """


@notebook_only
@code
def _add_tutorial_dir_to_path():
    import sys, os
    # `modal run` adds the script's directory to sys.path automatically;
    # notebooks don't, so the sibling `haiku` module isn't findable.
    if os.getcwd() not in sys.path:
        sys.path.insert(0, os.getcwd())


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

    SLIME reads training data from parquet files on a shared volume. A
    `DatasetConfig` subclass specifies two things:

    1. **Where the files live** (`prompt_data`, `eval_prompt_data`) —
       paths under `DATA_PATH` (`/data` inside every SLIME container).
    2. **How to produce them** (`prepare()`) — runs once inside the
       `prepare_dataset` Modal function with the data volume mounted
       read-write, and should write the parquet files atomically.

    The class attrs also map to SLIME CLI flags:

    - `input_key="messages"` — column name SLIME reads the chat
      conversation from.
    - `label_key="label"` — column holding the ground-truth poem (used
      only by the optional LLM judge to grade relative quality; the
      structure-only reward ignores it).
    - `apply_chat_template=True` — ask SLIME to run the tokenizer's
      chat template over `messages` before rollout, matching what the
      model was pretrained with.
    - `rm_type="async_rm"` — the reward model is an async Python
      function rather than a built-in (`math`, `deepscaler`, …) or a
      remote HTTP service.

    Inside `prepare()` we:

    1. Download
       [`statworx/haiku`](https://huggingface.co/datasets/statworx/haiku),
       a dataset of ~7k haiku poems tagged with a `keywords` topic.
    2. Turn each row into a chat conversation: a system prompt that
       primes the model to incorporate Modal vocabulary
       (`haiku.MODAL_VOCABS`, re-used between dataset prep and judge
       rubric) and a user prompt like `"Write me a haiku about cat."`.
    3. Also precompute the tokenized `prompt` string via the model's
       chat template — SLIME uses that as the exact rollout input.
    4. Hold out the last 20% (capped at 1000 rows) as a test split so
       `eval_prompt_data` has something to score against during
       training.
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
            # `data_path` is DATA_PATH on whichever container is mounting
            # the shared volume — /data on SLIME containers. Stashing it
            # plus the HF model id lets prepare() run standalone later.
            self._data_path = str(data_path)
            self._hf_checkpoint = hf_checkpoint
            self.prompt_data = f"{self._data_path}/haiku/train.parquet"
            self.eval_prompt_data = ["haiku", f"{self._data_path}/haiku/test.parquet"]

        def prepare(self):
            import os

            from datasets import load_dataset
            from transformers import AutoTokenizer

            # Use the base-model tokenizer to render the chat template;
            # SLIME will re-tokenize at rollout, but pre-rendering the
            # prompt string here lets us log exactly what the model sees.
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
                # statworx/haiku rows look like
                # {"keywords": "Cat", "text": "<reference haiku>"}.
                # We turn `keywords` into a user question and keep the
                # reference poem under `label` for the judge to consume.
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

            # Hold out the last 20% (≤ 1000 rows) as the test split so
            # `eval_prompt_data` has fresh prompts the policy never sees
            # during rollout.
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

    GRPO needs a scalar reward for every rollout. The sibling `haiku.py`
    module defines `haiku_rm`, an **async** reward function matching the
    signature SLIME expects:

    ```python
    async def haiku_rm(args, sample, **kwargs) -> float:
        ...
    ```

    - `args` is the SLIME CLI namespace (rarely needed).
    - `sample` exposes `.response` (the model's generation), `.prompt`
      / `.question` (the input), and `.label` (the reference answer,
      when available).
    - Return value is the reward. Higher is better; GRPO subtracts the
      group mean so absolute magnitudes are less important than
      consistency across samples.

    `haiku_rm` composes two scoring components:

    - **`score_haiku_structure(response, cmudict)`** — pure-Python check
      that returns a value in `[0, 1]`: a quarter-point for having
      three lines, plus up to a quarter per line for hitting the 5-7-5
      syllable target (half-credit for off-by-one). Uses NLTK's CMUdict
      for pronunciation; falls back to a vowel-group heuristic for
      out-of-dict words.
    - **`HaikuStyleJudge(LlmJudge)`** — subclass of the shared
      `modal_training_gym.common.llm_judge.LlmJudge`. The base class
      handles the boring part (POST to an OpenAI-compatible
      chat-completions endpoint, regex-parse a number out of the reply,
      normalize to `[0, 1]`). The subclass overrides `build_prompt()`
      with the haiku-specific rubric: relevance + poetic quality + Modal
      vocabulary usage.

    The two components are combined with **curriculum-style gating** —
    the style score is evaluated only when the structure score is
    perfect (`1.0`). Early in training the model has to learn 5-7-5
    first; once it does, the style score kicks in and rewards
    thematically interesting poems. Total reward is therefore
    `structure + style ∈ [0, 2]`.

    Gating also makes the style path **opt-in**: it only fires when
    `LLM_JUDGE_URL` is set in the training environment. Without a judge
    deployed, the reward degrades gracefully to structure-only and the
    run still progresses — convenient for smoke-testing this tutorial.
    """


@markdown
def _explain_config():
    """
    ## Define the experiment

    One big `SlimeConfig(...)` call. Every attribute except the
    launcher-only ones (`environment`, `async_mode`, `slime_model_script`,
    `dataset`, `model`, `wandb`, `modal`, `app_tags`) becomes a SLIME CLI
    flag via `field_name` → `--field-name`. Grouped below by concern.

    ### Model, cluster, and checkpoint format

    - `model=Qwen3_4B()` — attaches the shared `HFModelConfiguration`
      subclass; the launcher uses it for `download_model()` and expands
      its `ModelArchitecture` into per-layer SLIME flags (num_layers,
      hidden_size, etc.).
    - `ref_load=base_model.model_name` — GRPO needs a frozen reference
      policy for its KL term; load it from the same HF checkpoint.
    - `megatron_to_hf_mode="bridge"` — saves checkpoints through
      Megatron-Bridge so vLLM can load them directly as HF-format weights
      without a separate conversion step (`iter_NNNNNNN_hf/` dirs under
      `save`).
    - `actor_num_nodes=1`, `actor_num_gpus_per_node=8`, `colocate=True`
      — single 8×H200 node with rollout (sglang) and training sharing
      the same GPUs. Colocation keeps weight sync cheap; the tradeoff is
      memory pressure, which we manage with the sglang + recompute knobs
      below.

    ### Reward wiring

    - `rm_type="async_rm"` + `custom_rm_path="haiku.haiku_rm"` — SLIME
      imports `haiku` by module name and calls `haiku.haiku_rm(...)` per
      rollout sample.
    - `local_python_sources=["haiku"]` on `ModalConfig` — the slime
      launcher runs `image.add_local_python_source("haiku", copy=True)`
      at image-build time, which follows Python's normal import
      machinery to find `tutorials/rl/slime_haiku/haiku.py` on `sys.path`
      (put there automatically by `modal run path/to/slime_haiku.py`)
      and bakes it into the training image. On the remote container,
      `import haiku` then resolves under its plain top-level name — no
      PYTHONPATH surgery.

    ### Rollout and GRPO group shape

    - `num_rollout=50` — 50 training rollout passes (≈ one epoch here).
    - `rollout_batch_size=128` — 128 prompts sampled per pass.
    - `n_samples_per_prompt=8` — per-prompt group size for GRPO's
      within-group advantage.
    - `global_batch_size=64` — optimizer step batch size; each step
      consumes `global_batch_size` samples out of the
      `rollout_batch_size × n_samples_per_prompt = 1024` rollout buffer.
    - `rollout_max_response_len=300` — haiku are short; cap generation
      at 300 tokens.
    - `rollout_num_gpus_per_engine=2` — two sglang engines share the
      node (8 GPUs / 2 per engine → 4 engines), each running TP=2.

    ### Evaluation

    - `eval_interval=20` — evaluate every 20 training steps.
    - `n_samples_per_eval_prompt=8` — sample 8 completions per eval
      prompt for a stable success rate.

    ### Megatron training knobs

    `tensor_model_parallel_size=2`, `sequence_parallel=True`,
    `use_dynamic_batch_size=True` with `max_tokens_per_gpu=9216`, and
    `recompute_granularity="full"` + `recompute_method="uniform"` +
    `recompute_num_layers=1` together give a memory-for-time tradeoff
    that keeps the colocated actor + rollout + KL reference model
    within H200 HBM. The `accumulate_*_in_fp32` flags and
    `attention_softmax_in_fp32=True` are the usual RL numerics
    safeguards.

    ### Optimizer + GRPO algorithm

    - `lr=1e-6` with constant schedule — RL policy updates are fragile;
      small, flat learning rate.
    - `eps_clip=0.2`, `eps_clip_high=0.28` — GRPO's
      PPO-style ratio clip (asymmetric high bound is a common GRPO
      tweak; keeps updates stable on high-advantage tokens).
    - `use_kl_loss=True`, `kl_loss_coef=0.0`, `kl_loss_type="low_var_kl"`
      — register the KL penalty but keep its weight at zero; structure
      is enforced by the reward, not the KL.
    - `entropy_coef=0.0` — no explicit entropy bonus.

    ### Checkpointing + Modal image

    - `save="/checkpoints/qwen3-4b-haiku"`, `save_interval=10` —
      checkpoint every 10 steps into the `slime-checkpoints` volume; the
      **Serve** section below hosts one of those directories.
    - `ModalConfig(image_run_commands=[...])` — installs `aiohttp`
      (HTTP client for the LLM judge) and `nltk` (CMUdict) into the
      SLIME base image, plus pre-downloads `cmudict` at image-build time
      so the first reward call doesn't pay the download cost.
    """


@code
def _define_config():
    base_model = Qwen3_4B()
    my_training_run = SlimeConfig(
        # ── Model + frozen reference policy ────────────────────────────
        model=base_model,
        ref_load=base_model.model_name,
        megatron_to_hf_mode="bridge",

        # ── Shared containers ─────────────────────────────────────────
        dataset=HaikuDataset(DATA_PATH, base_model.model_name),
        wandb=WandbConfig(project="slime-grpo", group="qwen3-4b-haiku"),

        # ── Cluster shape ─────────────────────────────────────────────
        actor_num_nodes=1,
        actor_num_gpus_per_node=8,
        colocate=True,

        # ── Custom reward wiring ──────────────────────────────────────
        rm_type="async_rm",
        custom_rm_path="haiku.haiku_rm",

        # ── Rollout + GRPO group shape ────────────────────────────────
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

        # ── Evaluation ────────────────────────────────────────────────
        eval_interval=20,
        n_samples_per_eval_prompt=8,
        eval_max_response_len=300,
        eval_top_p=1.0,

        # ── Megatron training knobs ───────────────────────────────────
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

        # ── Optimizer ─────────────────────────────────────────────────
        optimizer="adam",
        lr=1e-6,
        lr_decay_style="constant",
        weight_decay=0.1,
        adam_beta1=0.9,
        adam_beta2=0.98,

        # ── GRPO algorithm knobs ──────────────────────────────────────
        advantage_estimator="grpo",
        eps_clip=0.2,
        eps_clip_high=0.28,
        use_kl_loss=True,
        kl_loss_coef=0.0,
        kl_loss_type="low_var_kl",
        entropy_coef=0.0,

        # ── Checkpointing ─────────────────────────────────────────────
        save="/checkpoints/qwen3-4b-haiku",
        save_interval=10,

        # ── Modal infrastructure ──────────────────────────────────────
        modal=ModalConfig(
            gpu="H200",
            # Ship the sibling reward module into the training image so
            # SLIME's custom_rm_path import resolves remotely.
            local_python_sources=["haiku"],
            # Runtime deps for the reward function: aiohttp for the LLM
            # judge client, nltk + cmudict for syllable counting.
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

    `build_app()` returns a `modal.App` with four functions defined
    against the right volumes, secrets, and GPU spec:

    - `download_model` — pulls the HF checkpoint into the
      `huggingface-cache` volume (1×H200, 2 hour timeout).
    - `prepare_dataset` — runs `HaikuDataset.prepare()` against the
      `slime-data` volume (CPU).
    - `convert_checkpoint` — no-op in bridge mode; for `megatron_to_hf_mode
      ="raw"` experiments it'd do the HF → torch_dist conversion.
    - `train` — the actual GRPO run on a Modal `@clustered(n_nodes)`
      cluster (1×8×H200 here), with Ray brought up in-container via
      `ModalRayCluster`.

    Every function closes over the `my_training_run` config above, so the
    app is fully determined by that single object.
    """


@code
def _build_app():
    app = my_training_run.build_app()


@markdown
def _run_section():
    """
    ## Run training
    """


@py_only
@markdown
def _run_cli():
    """
    From the CLI, run the three stages in order. `download_model` and
    `prepare_dataset` are one-shots — once the HF weights and parquet
    files are on their volumes, reruns skip the download. `train` is
    long-running; `--detach` keeps it going after you close the terminal:

    ```bash
    uv run modal run tutorials/rl/slime_haiku/slime_haiku.py::app.download_model
    uv run modal run tutorials/rl/slime_haiku/slime_haiku.py::app.prepare_dataset
    uv run modal run --detach tutorials/rl/slime_haiku/slime_haiku.py::app.train
    ```

    To enable the LLM-judge style score, set `LLM_JUDGE_URL` (and
    optionally `LLM_JUDGE_MODEL`) on the train function's environment —
    either by editing the `environment` dict in `ModalConfig`, or via a
    Modal secret attached to the `train` function. Omit it entirely for
    the structure-only variant that this tutorial ships with by default.
    """


@notebook_only
@markdown
def _run_interactive():
    """
    In a notebook, `app.run()` opens an ephemeral app context and
    `.remote()` invokes a remote function inside it. `modal.enable_output()`
    streams container logs into the notebook so you can watch the run.
    Put one stage per cell to checkpoint your progress:
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

    `build_vllm_serve_app(...)` is the post-training counterpart to
    `build_app()`: a thin wrapper around the canonical
    [Modal vLLM inference example](https://modal.com/docs/examples/vllm_inference)
    that returns a **second** Modal app, fully independent of the
    training app. It registers one `@modal.web_server`-decorated
    function that mounts the checkpoints volume, runs
    `vllm serve <model_path>`, and exposes the standard OpenAI
    chat-completions API at
    `https://<workspace>--<app_name>-serve.modal.run`.

    Because it's a separate app, you `modal deploy` it (not
    `modal run`) — that gives you a long-lived URL you can point a
    playground UI or eval harness at. Modal scales the replica to zero
    after `scaledown_window` of idleness.

    ### Pointing at the right checkpoint

    `model_path` must point at a directory containing an **HF-format**
    checkpoint (`config.json` plus weight shards). SLIME bridge mode
    writes Megatron torch_dist checkpoints under `save` — pick out an
    HF-compatible subdirectory once training completes, or run a
    torch_dist → HF conversion if your config didn't emit one. Inspect
    what the training left behind with:

    ```bash
    modal volume ls slime-checkpoints qwen3-4b-haiku/
    ```

    and point `model_path` at whichever directory has a `config.json`.
    The reference
    [qwen3-haiku `convert_torch_dist_to_hf.py`](https://github.com/modal-labs/qwen3-haiku/blob/main/tools/convert_torch_dist_to_hf.py)
    is a worked example of the conversion flow if you need it.
    """


@code
def _build_serve_app():
    serve_app = build_vllm_serve_app(
        # Modal app name — determines the deployed URL shape:
        # https://<workspace>--serve-haiku-model-serve.modal.run
        app_name="serve-haiku-model",
        # Absolute path inside the container to an HF-format checkpoint
        # directory (must contain config.json). Update after inspecting
        # the slime-checkpoints volume — the exact subdirectory depends
        # on slime's bridge-mode output layout. (To smoke-test the
        # serving pipeline before training, swap this for the HF repo
        # id "Qwen/Qwen3-4B" and drop the `checkpoints_volume` arg
        # below; vLLM will download the base model itself.)
        model_path="/checkpoints/qwen3-4b-haiku",
        # The `model` field callers pass in chat-completions requests.
        served_model_name="qwen3-4b-haiku",
        # Mount the training checkpoints volume; without this,
        # /checkpoints wouldn't exist in the serving container.
        checkpoints_volume="slime-checkpoints",
        gpu="H100",
        n_gpu=1,
        # Qwen3's reasoning-parser separates <think>...</think> traces
        # from the final response on the server side. --enforce-eager
        # skips CUDA-graph capture (faster cold starts, slightly slower
        # inference — ideal for scale-from-zero endpoints).
        extra_vllm_args=["--enforce-eager", "--reasoning-parser", "qwen3"],
    )


@py_only
@markdown
def _serve_cli():
    """
    Deploy the serving app separately from the training app:

    ```bash
    uv run modal deploy tutorials/rl/slime_haiku/slime_haiku.py::serve_app
    ```

    The URL is deterministic once deployed:
    `https://<workspace>--serve-haiku-model-serve.modal.run`. Query it
    with any OpenAI-compatible client — the endpoint speaks the
    standard chat-completions API:

    ```bash
    curl https://<workspace>--serve-haiku-model-serve.modal.run/v1/chat/completions \\
      -H "Content-Type: application/json" \\
      -d '{
            "model": "qwen3-4b-haiku",
            "messages": [{"role": "user", "content": "Write me a haiku about cat."}],
            "temperature": 0.0
          }'
    ```

    To swap checkpoints, edit `model_path` above and redeploy — Modal
    will spin up a fresh container pointed at the new weights.
    """


@notebook_only
@markdown
def _serve_interactive():
    """
    From a notebook, `serve_app.deploy()` does exactly what
    `modal deploy` does — creates a long-lived deployment rather than an
    ephemeral run. `modal.enable_output()` streams the deploy logs so
    you can see when vLLM finishes loading the weights.
    """


@notebook_only
@code
def _invoke_deploy_serve():
    with modal.enable_output():
        serve_app.deploy()
