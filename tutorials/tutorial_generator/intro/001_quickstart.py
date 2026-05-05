"""Tutorial source for `001_quickstart` — parsed by generate_tutorial.py.

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
    'api_classes': [
        'ModelConfig', 'HFModelConfiguration', 'ModelArchitecture',
        'Qwen3_0_6B', 'DatasetConfig', 'WandbConfig', 'TrainConfig', 'SlimeRecipe',
    ],
}


from tutorial_generator import code, markdown, notebook_only, py_only, shell


@markdown
def _intro():
    """
    # Quickstart: concepts that every tutorial uses

    `modal-training-gym` wraps training frameworks (slime)
    behind a single pattern: declare
    a model, a dataset, and a logging config; hand
    them to a framework factory; `modal run` the returned app. This
    notebook walks through the pieces so the per-framework tutorials can
    focus on what makes each framework different instead of re-explaining
    the plumbing.

    Nothing here launches a job. At the end there are links to concrete
    tutorials at each difficulty tier.
    """


@notebook_only
@shell("%uv pip install -q git+https://github.com/modal-projects/training-gym.git@main")
def _install():
    pass


@markdown
def _containers_section():
    """
    ## The three config containers

    Every tutorial builds its run from three pure-data containers: a
    `ModelConfig`, a `DatasetConfig`, and a `WandbConfig`. Each
    framework's launcher translates these into its own CLI vocabulary —
    you don't write framework-specific flag names for the parts that are
    shared across frameworks.
    """


@markdown
def _model_container():
    """
    ### `ModelConfig` — the model to train

    A `ModelConfig` is identity + download hook. Built-in subclasses
    (`Qwen3_0_6B`, `Qwen3_4B`, `Qwen3_32B`, `GLM_4_7`, `Llama2_7B`,
    `KimiK2_5`) set
    `model_name` and a `ModelArchitecture` spec. `HFModelConfiguration` is
    the base for any HF-hosted model — it implements `download()` via
    `huggingface_hub.snapshot_download` pulling weights into the
    `huggingface-cache` Modal volume.

    For a custom HF model, subclass `ModelConfig` inline in your
    tutorial — there's no registry to update.
    """


@code
def _model_example():
    from modal_training_gym.common.models import Qwen3_0_6B

    model = Qwen3_0_6B()
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
    that takes a framework-specific config and returns a `modal.App`
    with a `train` function. Calling `train` handles everything:
    downloading the model (if not already cached), preparing the
    dataset (if not already materialized), and running the training
    job. One call, one command.

    The shape, written out:

    ```python
    from modal_training_gym.common.models import Qwen3_0_6B
    from modal_training_gym.common.train import TrainConfig
    from modal_training_gym.common.wandb import WandbConfig
    from modal_training_gym.train_recipes.slime_recipe import SlimeRecipe

    run = TrainConfig(
        model=Qwen3_0_6B(),
        dataset=MyDataset(...),
        recipe=SlimeRecipe(
            wandb=WandbConfig(project="my-runs", group="concepts-demo"),
            # … framework-specific flags …
        ),
    )

    app = run.build_app()   # modal.App — call app.train to run everything
    ```

    The framework-specific flags differ across frameworks,
    but everything around them stays put.
    """


@markdown
def _volumes_section():
    """
    ## Volume layout

    Three Modal volumes are shared across tutorials so you pay the
    download cost once:

    | Path (in container) | Volume | Contents |
    |---|---|---|
    | `/root/.cache/huggingface` | `huggingface-cache` | HF weights + tokenizers. Populated by `download`. |
    | `/data` | `<framework>-data` | Preprocessed datasets. Populated by `prepare_dataset`. |
    | `/checkpoints` | `<framework>-checkpoints` | Training outputs. Populated by `train`. |

    Each framework uses its own data and checkpoints volumes (so one
    framework's half-written state can't corrupt another's), but the HF
    cache is truly shared — download `qwen3-4b` once, use it from any
    framework.

    Nothing in a tutorial directly manipulates volumes — the launchers
    mount them and the three remote functions know where to write. If you
    need to delete stale checkpoints, use `modal volume rm` against the
    relevant volume from a shell.
    """


@markdown
def _running_section():
    """
    ## Running a training job

    Every tutorial has one command to run. `train` handles model
    download, dataset preparation, and training in a single call.
    Always `--detach` so the run survives your terminal closing:

    ```bash
    uv run modal run --detach tutorials/<tutorial>/<tutorial>.py::app.train
    ```

    Reattach to a detached run's logs from another terminal with
    `modal app logs <app-name>`.

    **Interactive** — inside a notebook cell, open an ephemeral app
    and call `train` directly:

    ```python
    with modal.enable_output():
        with app.run():
            app.train.remote()
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
    - [`002_custom_model`](../../intro/002_custom_model/002_custom_model.ipynb) —
      serve and eval a custom HuggingFace model (SmolLM2-135M) with an
      inline `ModelConfig` subclass, no training.
    **Intermediate** (non-default wiring, 1–2 nodes)

    - [`slime_haiku`](../../rl/slime_haiku/slime_haiku.ipynb) — GRPO with a
      **custom async reward function** (structure scoring + optional LLM
      judge).
    - [`ray_slime_standalone`](../../misc/ray_slime_standalone/ray_slime_standalone.ipynb) —
      Ray-on-Modal pattern demo with a custom training loop.

    **Advanced** (≥2 nodes, non-trivial parallelism, large models)

    - [`slime_gsm8k`](../../rl/slime_gsm8k/slime_gsm8k.ipynb) — colocated 4-node
      GRPO, the canonical math-RL reference.
    """
