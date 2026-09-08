# Agent Guide: Training Gym Overview

One-stop reference for agents asked to build, modify, or validate tutorials
and examples in this repo. Pairs with
[modal-infrastructure](modal-infrastructure/SKILL.md) (Modal launch/debug) and
[agent-example-validation.md](agent-example-validation.md) (tiered example
validation).

## What this repo is

`modal-training-gym` is a pip-installable Python package plus a catalog of
runnable training tutorials on Modal's multi-node GPU cluster product. End
users `pip install modal-training-gym` once, then import framework-specific
launchers from their own scripts or notebooks.

## Top-level layout

```
modal_training_gym/         ← installable package
├── common/                 ← cross-framework pure data + helpers
│   ├── dataset.py          ← DatasetConfig base (user subclasses)
│   ├── models/             ← ModelConfig hierarchy (see below)
│   ├── wandb.py            ← WandbConfig
│   ├── framework.py        ← resolve_caller_module, TOOLS_*
│   └── ray_cluster.py      ← ModalRayCluster helper (used by slime)
├── frameworks/             ← one subpackage per training framework
│   ├── slime/              ← slime GRPO (Ray + Megatron + SGLang)
└── tools/                  ← shared scripts mounted on every image at
                              /opt/training-gym/tools (see "Tools" below)

tutorials/                  ← flat runnable Python tutorial sources

tests/                      ← plain-script tests (uv run tests/<x>.py)
skills/                     ← agent-facing docs (you are here)
```

Edit `tutorials/*.py` or `tutorials/<name>/main.py` directly; each entry is
the runnable tutorial and the docs source, with sibling helpers beside
`main.py`.

## Core abstractions

### `ModelConfig` / `HFModelConfiguration`

Model identity + optional architecture + optional local path + a
`download()` method. Found in `modal_training_gym/common/models/`.

```python
class ModelConfig:                       # base; abstract download
    model_name: str = ""
    model_path: str | None = None
    architecture: ModelArchitecture | None = None
    def __init__(self, **kwargs): ...
    def download(self): raise NotImplementedError

class HFModelConfiguration(ModelConfig): # HF-hosted; snapshot_download
    def download(self):
        snapshot_download(repo_id=self.model_name)
```

Built-in subclasses:

| Class | HF repo | Architecture populated? | Notes |
|---|---|---|---|
| `Qwen3_4B` | `Qwen/Qwen3-4B` | yes | slime-ready |
| `GLM_4_7` | `zai-org/GLM-4.7` | no | architecture inferred from HF config |
| `Llama2_7B` | `meta-llama/Llama-2-7b-hf` | no | torchrun-based workflows |

**Rule of thumb for slime**: slime emits architecture fields as CLI flags,
so it requires `architecture` to be a populated `ModelArchitecture(...)`.
The `SlimeConfig._validate_custom_model_architecture` preflight raises an
actionable `ValueError` if a user attaches a model with
`architecture is None`. Every other framework only needs `model_name`.

### `DatasetConfig`

In `modal_training_gym/common/dataset.py`. Plain class; subclass and override
`prepare()` to materialize the data on a shared volume. Declarative class
attrs (`prompt_data`, `input_key`, `rm_type`, etc.) are interpreted by each
framework's config converter.

### Framework config two-class split

Every framework exposes **two** dataclasses:

- `<F>FrameworkConfig` — Modal infra (gpu, image, n_nodes, gpus_per_node) +
  framework-specific CLI flags. Uses pydantic with `extra="forbid"`, so any
  unknown kwarg fails loudly.
- `<F>Config` — wraps `dataset`, `model`, `metrics`, `framework_config`.
  Exposes `build_app()` which delegates to the launcher's
  `build_<f>_app(...)` factory. Typically has `_WRAPPER_FIELDS` (in some
  frameworks) to exclude the wrapper slots from CLI-arg rendering.

User code builds an app like:

```python
cfg = MyFrameworkConfig(
    dataset=MyDataset(...),
    model=Qwen3_4B(),
    metrics=WandbConfig(project="..."),
    framework_config=MyFrameworkFrameworkConfig(gpu="H100", n_nodes=1, ...),
)
app = cfg.build_app()
```

### `build_app()` delegation

`<F>Config.build_app()` → `build_<f>_app(<f>=self)` → constructs a
`modal.App` with Modal functions for each stage (typically
`download`, `prepare_dataset`, `train` / `train_multi_node`, plus
framework-specific ones like `convert_hf_to_mcore`, `upload_reward`, etc).

The launcher walks the call stack via
`common.framework.resolve_caller_module()` to find the true user-tutorial
module (skipping `modal_training_gym.*` frames) and registers that module
for cloudpickle by-value inlining — this is how a user's inline
`DatasetConfig` / `ModelConfig` subclasses survive serialization to
the remote container.

## Framework catalog

| Framework | Uses `config.model` CLI-wise | Needs `architecture` | Best for |
|---|---|---|---|
| `slime` | hf_checkpoint CLI flag **plus** architecture flags | **yes** | GRPO |

## The `tools/` shared directory

Any cross-framework script lives at `modal_training_gym/tools/`. Every
launcher mounts this directory at **`/opt/training-gym/tools`** on its
remote image(s) via `common.framework.mount_tools_dir`, so scripts are at a
predictable path regardless of which framework's container calls them.
Framework-agnostic `ModelConfig.download` overrides use this path.

To add a new tool: drop the script in `modal_training_gym/tools/`, commit.
It's automatically mounted via `add_local_dir(TOOLS_LOCAL_PATH,
remote_path=TOOLS_REMOTE_PATH, copy=True)` on every framework image.

## Adding a new model

1. **Create the per-model module** at
   `modal_training_gym/common/models/<name>.py`:

   ```python
   from .base import HFModelConfiguration, ModelArchitecture

   class MyModel(HFModelConfiguration):
       model_name = "org/repo"
       # Optional: populate only if slime users will consume this model.
       architecture = ModelArchitecture(
           num_layers=..., hidden_size=..., ...,
       )
       # Optional: override download only if the default HF
       # snapshot_download isn't enough.
   ```

2. **Export in `common/models/__init__.py`**:

   ```python
   from .my_model import MyModel
   __all__ = [..., "MyModel"]
   ```

3. **Verify** via `tests/test_model_configuration.py` (or an analogous
   snippet):

   ```python
   from modal_training_gym.common.models import MyModel
   m = MyModel()
   assert m.model_name == "org/repo"
   ```

### When to override `download`

- Just HF snapshot → inherit `HFModelConfiguration` (do nothing).
- Extra post-processing (format conversion, weight repacking, tokenizer
  tweaks) → override `download` in the subclass. Reference
  `tools/<script>.py` via the canonical `/opt/training-gym/tools` path.
  Do **not** put this logic in a framework launcher — keep model-specific
  quirks with the model spec, not the framework plumbing.

## Adding a new tutorial

1. **Pick the framework**. Use one of the catalog entries above in most cases.

2. **Create `tutorials/<name>.py`** with the next contiguous `order` value:

   ```python
   # ---
   # order: 0
   # ---
   #
   # # <Title>
   #
   # One-paragraph description of what this trains.

   from modal_training_gym import TrainConfig

   train_result = TrainConfig(...).train()
   ```

   Markdown comment blocks become prose on the docs page. Python blocks become
   code cells.

3. **Validate the source**:

   ```bash
   uv run -m compileall tutorials/
   ```

4. **Keep training cheap by default**. Tutorials should smoke in a single
   step by default so Tier 2 validation is cheap:
   - `num_train_epochs=1`, `train_iters=1` (or framework equivalent).
   - Small `global_batch_size` / tiny dataset slice
     (e.g. `split="train[:4]"`).
   - `log_interval=1` so the first training step is visible.
   - Disable eval / save if they'd gate the first-step marker
     (`test_freq=-1`, `save_freq=-1`).

### Custom-model tutorial pattern (no built-in subclass)

If the tutorial's model isn't in the catalog, define a one-off subclass
inline:

```python
from huggingface_hub import snapshot_download
from modal_training_gym import ModelArchitecture, ModelConfig


class MyTinyModel(ModelConfig):
    model_name = "HuggingFaceTB/SmolLM2-135M"
    # Slime also requires a populated model architecture.
    architecture = ModelArchitecture(...)

    def download(self):
        snapshot_download(repo_id=self.model_name)


model = MyTinyModel()
```

Better: inherit from `HFModelConfiguration` and skip the `download`
override (the one-line snapshot_download is the inherited default). See
existing slime tutorials for full examples.

## Validation

Always follow the tiered policy in
[agent-example-validation.md](agent-example-validation.md):

- **Tier 0 (local compile)** — `uv run -m compileall modal_training_gym/`.
- **Tier 1 (cheap drift checks):** Local instantiation smoke across the
  affected frameworks. No GPU.
- **Tier 2 (scheduled smoke)** — one remote `modal run --detach` that
  reaches ≥1 training step, then kill the detached app.
- **Tier 3 (full example validation)** — canonical multi-node runs.
  Scheduled, not per-PR gating.

Per-change default: Tier 0 + Tier 1, plus Tier 2 for the new/modified
tutorial only. Don't expand to all tutorials on a single change.

`tests/test_model_configuration.py` is the current model-API regression
test. Run with `uv run tests/test_model_configuration.py`.

## Gotchas

- **Python version pin**. The repo pins 3.12 (see `pyproject.toml` +
  `CLAUDE.md`). Modal's `serialized=True` functions require the remote
  image's Python to match the local one. If a framework image has Python
  3.11 (e.g. some ModelScope images), app build fails with `InvalidError`.
- **Framework image switches**. To override a framework's default image,
  set `image=` on `<F>FrameworkConfig` in your tutorial. The launcher's
  `pip_install` chain reinstalls the framework fresh, so
  switching the base is usually enough. Check whether transitive deps
  (megatron-core, pillow, tokenizers for transformers) are in the new
  image; the ModelScope image shipped many, bare CUDA/NGC images don't.
- **cloudpickle caller_module**. `SlimeConfig.build_app()` (and the other
  wrapper `build_app` methods) now all delegate to the launcher, meaning
  `inspect.stack()[1]` inside `build_<f>_app` is the config wrapper, not
  the tutorial. Launchers use `resolve_caller_module()` to walk past
  `modal_training_gym.*` frames. Never use raw `inspect.stack()[1]` here.
- **Secrets for gated models and W&B**. Hugging Face auth is only needed for
  gated or rate-limited Hub access. Pass `WandbConfig` only when you want W&B.
- **Do not add framework-specific quirks to `<F>Config`** that only matter
  for one model. Put those in the model's `download` override and
  make the tool script live in `modal_training_gym/tools/`.

## Common file references

- Adding/modifying a model → `modal_training_gym/common/models/`.
- Adding/modifying a framework → `modal_training_gym/frameworks/<name>/`.
- Cross-framework scripts → `modal_training_gym/tools/`.
- Cross-framework helpers → `modal_training_gym/common/framework.py`.
- Tutorial sources live in `tutorials/*.py` or `tutorials/<name>/main.py`;
  their docs loader is `docs-next/src/lib/tutorial-docs-loader.ts`.
- Tests → `tests/test_*.py`, run via `uv run tests/<file>.py`.
