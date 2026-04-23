"""Tutorial source for `quickstart` — parsed by generate_tutorial.py.

Concept-only walkthrough of the shared abstractions every framework tutorial
uses: the three config containers (Model, DatasetConfig, WandbConfig), the
framework factory pattern, the volume layout, and how to run the pipeline.
Read this first; then pick a framework tutorial.
"""

TUTORIAL_METADATA = {
    'framework': '— (concepts)',
    'cluster_shape': '—',
    'summary': 'Shared concepts: config containers, framework factories, volume layout, running the pipeline',
    'difficulty': 'Beginner',
    'order': 0,
}


from tutorial_generator import code, markdown, notebook_only, py_only, shell


@markdown
def _intro():
    """
    # Quickstart: concepts that every tutorial uses

    `modal-training-gym` wraps a handful of training frameworks (SLIME,
    Megatron, MS-SWIFT) behind a single pattern: declare
    a model, a dataset, and a logging config; hand
    them to a framework factory; `modal run` the returned app. This
    notebook walks through the pieces so the per-framework tutorials can
    focus on what makes each framework different instead of re-explaining
    the plumbing.

    Nothing here launches a job. At the end there are links to concrete
    tutorials at each difficulty tier.
    """


@notebook_only
@shell("%uv pip install -q git+https://github.com/modal-projects/training-gym.git@joy/initial-setup")
def _install():
    pass


@markdown
def _containers_section():
    """
    ## The three config containers

    Every tutorial builds its run from three pure-data containers: a
    `ModelConfiguration`, a `DatasetConfig`, and a `WandbConfig`. Each
    framework's launcher translates these into its own CLI vocabulary —
    you don't write framework-specific flag names for the parts that are
    shared across frameworks.
    """


@markdown
def _model_container():
    """
    ### `ModelConfiguration` — the model to train

    A `ModelConfiguration` is identity + download hook. Built-in subclasses
    (`Qwen3_4B`, `Qwen3_32B`, `GLM_4_7`, `Llama2_7B`, `KimiK2_5`) set
    `model_name` and a `ModelArchitecture` spec. `HFModelConfiguration` is
    the base for any HF-hosted model — it implements `download_model()` via
    `huggingface_hub.snapshot_download` pulling weights into the
    `huggingface-cache` Modal volume.

    For a custom HF model, subclass `ModelConfiguration` inline in your
    tutorial (see
    [`ms_swift_custom_hf`](../../sft/ms_swift_custom_hf/ms_swift_custom_hf.ipynb)) —
    there's no registry to update.
    """


@code
def _model_example():
    from modal_training_gym.common.models import Qwen3_4B

    model = Qwen3_4B()
    print("model_name:", model.model_name)
    print("num_layers:", model.architecture.num_layers)


@markdown
def _dataset_container():
    """
    ### `DatasetConfig` — what to train on

    Subclass `DatasetConfig`, set the fields your framework needs, and
    implement `prepare()` to materialize the data into the shared `/data`
    volume. `prepare()` runs inside a Modal container so it can import
    `datasets`, talk to HF Hub, or run any preprocessing; the output lands
    on a volume that every subsequent run (on any node) reads from.

    Fields like `input_key`, `label_key`, `apply_chat_template`, and
    `rm_type` are read by each framework's launcher and translated into its
    own flags — you only write them once.
    """


@code
def _dataset_example():
    from modal_training_gym.common.dataset import DatasetConfig

    class MyDataset(DatasetConfig):
        input_key = "messages"
        label_key = "label"
        apply_chat_template = True

        def __init__(self, data_path):
            self._data_path = str(data_path)
            self.prompt_data = f"{self._data_path}/my_dataset/train.parquet"

        def prepare(self):
            from datasets import load_dataset
            ds = load_dataset("some-org/my-dataset")
            ds["train"].to_parquet(self.prompt_data)

    ds = MyDataset(data_path="/data")
    print("prompt_data:", ds.prompt_data)
    print("input_key:", ds.input_key)


@markdown
def _wandb_container():
    """
    ### `WandbConfig` — where to log

    Just metadata: `project`, `group`, optional `exp_name`. The W&B API key
    is injected at launch time from the `wandb` Modal secret — you don't
    put it in code. Create the secret once via the Modal dashboard or
    `modal secret create`.
    """


@code
def _wandb_example():
    from modal_training_gym.common.wandb import WandbConfig

    wandb = WandbConfig(project="my-runs", group="concepts-demo")
    print("project:", wandb.project)


@markdown
def _factory_section():
    """
    ## The framework factory pattern

    Every framework package exposes a `build_<name>_app(...)` factory
    that takes a framework-specific config plus a `ModalConfig` and
    returns a `modal.App` with a standard set of remote functions:

    - `download_model` — pulls the model into the `huggingface-cache`
      volume. One-time per `(model,)` pair — subsequent runs reuse the
      volume.
    - `prepare_dataset` — runs your `DatasetConfig.prepare()` against the
      `/data` volume. One-time per `(dataset,)` pair.
    - `train` — the actual training run. This is the long one; launch it
      with `modal run --detach`.

    Framework-specific extras exist too: raw-mode SLIME/Megatron runs add
    `convert_checkpoint`; RL frameworks may add serving helpers. A
    tutorial's `.py` file is where all of this is wired together —
    calling `framework_config.build_app()` closes over your config and
    hands back the app.

    The same shape, written out:

    ```python
    from modal_training_gym.common.models import Qwen3_4B
    from modal_training_gym.common.wandb import WandbConfig
    from modal_training_gym.frameworks.slime import (
        ModalConfig, SlimeConfig,
    )

    run = SlimeConfig(
        model=Qwen3_4B(),
        dataset=MyDataset(...),
        wandb=WandbConfig(project="my-runs", group="concepts-demo"),
        # … framework-specific flags …
        modal=ModalConfig(gpu="H200"),
    )

    app = run.build_app()   # modal.App with download_model / prepare_dataset / train
    ```

    Different framework, same shape — swap `slime` for `ms_swift`,
    `megatron`, etc. The framework-specific flags differ (that's what makes
    the frameworks different), but everything around them stays put.
    """


@markdown
def _volumes_section():
    """
    ## Volume layout

    Three Modal volumes are shared across tutorials so you pay the
    download cost once:

    | Path (in container) | Volume | Contents |
    |---|---|---|
    | `/root/.cache/huggingface` | `huggingface-cache` | HF weights + tokenizers. Populated by `download_model`. |
    | `/data` | `<framework>-data` | Preprocessed datasets. Populated by `prepare_dataset`. |
    | `/checkpoints` | `<framework>-checkpoints` | Training outputs. Populated by `train`. |

    Each framework uses its own data and checkpoints volumes (so one
    framework's half-written state can't corrupt another's), but the HF
    cache is truly shared — download `Qwen3-4B` once, use it from slime,
    megatron, and ms-swift.

    Nothing in a tutorial directly manipulates volumes — the launchers
    mount them and the three remote functions know where to write. If you
    need to delete stale checkpoints, use `modal volume rm` against the
    relevant volume from a shell.
    """


@markdown
def _running_section():
    """
    ## Running the pipeline

    Every tutorial exposes the same three-stage pipeline. Two ways to
    invoke it:

    **CLI** — one command per stage, easy to script. Always `--detach` for
    training so the run survives your terminal closing.

    ```bash
    uv run modal run tutorials/<tutorial>/<tutorial>.py::app.download_model
    uv run modal run tutorials/<tutorial>/<tutorial>.py::app.prepare_dataset
    uv run modal run --detach tutorials/<tutorial>/<tutorial>.py::app.train
    ```

    Reattach to a detached run's logs from another terminal with
    `modal app logs <app-name>`.

    **Interactive** — inside a notebook cell, open an ephemeral app and
    run one stage at a time. Good for iterating on `prepare()` or on a
    config without re-submitting the whole script.

    ```python
    with modal.enable_output():
        with app.run():
            app.download_model.remote()
    ```

    The `modal.enable_output()` context manager streams logs back into
    the notebook so you can see what's happening in real time.
    """


@markdown
def _next_section():
    """
    ## Where to go next

    Pick a tutorial that matches what you're trying to learn. See the full
    catalog in [`tutorials/README.md`](../README.md).

    **Beginner** (single node, one concept at a time)

    - [`nccl_benchmark`](../../misc/nccl_benchmark/nccl_benchmark.ipynb) —
      validate that multi-node NCCL works in your workspace before
      running real training.
    - [`ms_swift_custom_hf`](../../sft/ms_swift_custom_hf/ms_swift_custom_hf.ipynb) —
      LoRA SFT on a tiny SmolLM2-135M, with an inline custom
      `ModelConfiguration` subclass.
    **Intermediate** (non-default wiring, 1–2 nodes)

    - [`slime_haiku`](../../rl/slime_haiku/slime_haiku.ipynb) — GRPO with a
      **custom async reward function** (structure scoring + optional LLM
      judge).
    - [`ray_slime_standalone`](../../misc/ray_slime_standalone/ray_slime_standalone.ipynb) —
      Ray-on-Modal pattern demo with a custom training loop.

    **Advanced** (≥2 nodes, non-trivial parallelism, large models)

    - [`slime_gsm8k`](../../rl/slime_gsm8k/slime_gsm8k.ipynb) — colocated 4-node
      GRPO, the canonical math-RL reference.
    - [`ms_swift_glm_4_7_gsm8k`](../../sft/ms_swift_glm_4_7_gsm8k/ms_swift_glm_4_7_gsm8k.ipynb) —
      GLM-4.7 LoRA SFT on GSM8K using ms-swift + Megatron.
    - [`megatron_glm_4_7_longmit128k`](../../sft/megatron_glm_4_7_longmit128k/megatron_glm_4_7_longmit128k.ipynb) —
      GLM-4.7 long-context LoRA on the NeMo bridge.
    """
