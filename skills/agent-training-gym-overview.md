# Agent Guide: Training Gym Overview

One-stop reference for agents asked to build, modify, or validate tutorials
and examples in this repo. Pairs with
[agent-modal-training.md](agent-modal-training.md) (Modal launch/debug) and
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
│   ├── models/             ← ModelConfiguration hierarchy (see below)
│   ├── wandb.py            ← WandbConfig
│   ├── framework.py        ← resolve_caller_module, mount_tools_dir, TOOLS_*
│   └── ray_cluster.py      ← ModalRayCluster helper (used by slime, miles)
├── frameworks/             ← one subpackage per training framework
│   ├── slime/              ← SLIME GRPO (Ray + Megatron + SGLang)
│   ├── verl/               ← verl RLVR (Ray + Megatron + vLLM)
│   ├── megatron/           ← NeMo Megatron LoRA
│   ├── ms_swift/           ← ms-swift Megatron SFT
│   ├── miles/              ← Miles RL (Ray + Megatron)
│   ├── hf_accelerate/      ← HF Accelerate (arbitrary train script)
│   ├── torchrun/           ← torchrun  (arbitrary train script)
│   └── lightning/          ← PyTorch Lightning Fabric
└── tools/                  ← shared scripts mounted on every image at
                              /opt/training-gym/tools (see "Tools" below)

tutorials/
├── tutorial_generator/     ← decorator-annotated source files — THIS is
│                             what you edit; each file is one tutorial
└── generate_tutorial.py    ← AST-walks each source, emits
                              tutorials/<name>/<name>.py + .ipynb

tests/                      ← plain-script tests (uv run python tests/<x>.py)
skills/                     ← agent-facing docs (you are here)
```

**Never edit `tutorials/<name>/<name>.py` or `.ipynb` directly — they are
generated.** Edit `tutorials/tutorial_generator/<name>.py` and run
`uv run python tutorials/generate_tutorial.py`.

## Core abstractions

### `ModelConfiguration` / `HFModelConfiguration`

Model identity + optional architecture + optional local path + a
`download_model()` method. Found in `modal_training_gym/common/models/`.

```python
class ModelConfiguration:                       # base; abstract download_model
    model_name: str = ""
    model_path: str | None = None
    architecture: ModelArchitecture | None = None
    def __init__(self, **kwargs): ...
    def download_model(self): raise NotImplementedError

class HFModelConfiguration(ModelConfiguration): # HF-hosted; snapshot_download
    def download_model(self):
        snapshot_download(repo_id=self.model_name)
```

Built-in subclasses:

| Class | HF repo | Architecture populated? | Notes |
|---|---|---|---|
| `Qwen3_4B` | `Qwen/Qwen3-4B` | yes | SLIME-ready |
| `Qwen3_32B` | `Qwen/Qwen3-32B` | no (stub) | verl derives arch from HF config |
| `GLM_4_7` | `zai-org/GLM-4.7` | no | ms-swift / Megatron derive arch |
| `Llama2_7B` | `meta-llama/Llama-2-7b-hf` | no | torchrun / accelerate |
| `Kimi_K2_5` | `moonshotai/Kimi-K2.5` | no | **overrides `download_model`**: snapshot + INT4→BF16 conversion via `tools/convert_kimi_int4_to_bf16.py` |

**Rule of thumb for SLIME**: SLIME emits architecture fields as CLI flags,
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
- `<F>Config` — wraps `dataset`, `model`, `wandb`, `framework_config`.
  Exposes `build_app()` which delegates to the launcher's
  `build_<f>_app(...)` factory. Typically has `_WRAPPER_FIELDS` (in some
  frameworks) to exclude the wrapper slots from CLI-arg rendering.

User code builds an app like:

```python
cfg = MyFrameworkConfig(
    dataset=MyDataset(...),
    model=Qwen3_4B(),
    wandb=WandbConfig(project="..."),
    framework_config=MyFrameworkFrameworkConfig(gpu="H100", n_nodes=1, ...),
)
app = cfg.build_app()
```

### `build_app()` delegation

`<F>Config.build_app()` → `build_<f>_app(<f>=self)` → constructs a
`modal.App` with Modal functions for each stage (typically
`download_model`, `prepare_dataset`, `train` / `train_multi_node`, plus
framework-specific ones like `convert_hf_to_mcore`, `upload_reward`, etc).

The launcher walks the call stack via
`common.framework.resolve_caller_module()` to find the true user-tutorial
module (skipping `modal_training_gym.*` frames) and registers that module
for cloudpickle by-value inlining — this is how a user's inline
`DatasetConfig` / `ModelConfiguration` subclasses survive serialization to
the remote container.

## Framework catalog

| Framework | Uses `config.model` CLI-wise | Needs `architecture` | Best for |
|---|---|---|---|
| `slime` | hf_checkpoint CLI flag **plus** architecture flags | **yes** | GRPO |
| `verl` | hf_checkpoint (dir path) | no | RLVR (GRPO) |
| `megatron` | hf_checkpoint passed to train script | no (uses `AutoBridge`) | NeMo Megatron LoRA |
| `ms_swift` | `--model <model_name>` | no | ms-swift Megatron SFT |
| `miles` | `--hf-checkpoint` on the Miles argv | no | Miles RL |
| `hf_accelerate` | decorative (not dereferenced) | no | arbitrary train script |
| `torchrun` | decorative (not dereferenced) | no | arbitrary train script |
| `lightning` | decorative (not dereferenced) | no | arbitrary train script |

## The `tools/` shared directory

Any cross-framework script lives at `modal_training_gym/tools/`. Every
launcher mounts this directory at **`/opt/training-gym/tools`** on its
remote image(s) via `common.framework.mount_tools_dir`, so scripts are at a
predictable path regardless of which framework's container calls them.
Framework-agnostic `ModelConfiguration.download_model` overrides (e.g.
`Kimi_K2_5`) use this path.

Current tools:

- `convert_kimi_int4_to_bf16.py` — INT4 → BF16 conversion for Kimi K2.5.

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
       # Optional: populate only if SLIME users will consume this model.
       architecture = ModelArchitecture(
           num_layers=..., hidden_size=..., ...,
       )
       # Optional: override download_model only if the default HF
       # snapshot_download isn't enough (e.g. Kimi_K2_5 needs a post-
       # conversion step).
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

### When to override `download_model`

- Just HF snapshot → inherit `HFModelConfiguration` (do nothing).
- Extra post-processing (format conversion, weight repacking, tokenizer
  tweaks) → override `download_model` in the subclass. Reference
  `tools/<script>.py` via the canonical `/opt/training-gym/tools` path.
  Do **not** put this logic in a framework launcher — it keeps Kimi-style
  quirks with the model spec, not the framework plumbing.

## Adding a new tutorial

1. **Pick the framework** — almost always one of the catalog above.

2. **Create the source** at
   `tutorials/tutorial_generator/<name>.py`. Structure (follow
   `ms_swift_glm_4_7_gsm8k.py` or `ms_swift_custom_hf.py` as templates):

   ```python
   from tutorial_generator import code, markdown, notebook_only, py_only, shell


   @markdown
   def _intro():
       """# <Title>

       One-paragraph description of what this trains.
       """


   @notebook_only
   @shell("%uv pip install -q git+https://github.com/modal-projects/training-gym.git@joy/initial-setup")
   def _install():
       pass


   @code
   def _imports():
       import modal

       from modal_training_gym.common.dataset import DatasetConfig
       from modal_training_gym.common.models import <BuiltinModelOrModelConfiguration>
       from modal_training_gym.common.wandb import WandbConfig
       from modal_training_gym.frameworks.<framework> import (
           <F>Config, <F>FrameworkConfig,
       )


   @code
   def _define_dataset():
       class MyDataset(DatasetConfig):
           ...
           def prepare(self): ...


   @code
   def _define_config():
       fw_cfg = <F>FrameworkConfig(gpu="H100", n_nodes=1, gpus_per_node=1, ...)
       my_training_run = <F>Config(
           dataset=MyDataset(...),
           model=<BuiltinModel>(),
           wandb=WandbConfig(project="..."),
           framework_config=fw_cfg,
       )


   @code
   def _build_app():
       app = my_training_run.build_app()


   @py_only
   @markdown
   def _run_cli():
       """```bash
       uv run modal run tutorials/<name>/<name>.py::app.download_model
       uv run modal run tutorials/<name>/<name>.py::app.prepare_dataset
       uv run modal run --detach tutorials/<name>/<name>.py::app.train
       ```"""


   @notebook_only
   @code
   def _invoke_train():
       with modal.enable_output():
           with app.run():
               app.train.remote()
   ```

3. **Decorators cheat sheet**:
   - `@markdown` — function docstring becomes a markdown cell.
   - `@code` — function body (dedented) becomes a code cell.
   - `@shell("...")` — string arg is the code cell verbatim (supports
     `%uv pip install`, etc).
   - `@py_only` / `@notebook_only` — restrict a cell to one output format.
     Stack on top of `@markdown` / `@code` / `@shell`.

4. **Regenerate** and verify determinism:
   ```bash
   uv run python tutorials/generate_tutorial.py
   # Run it again — should produce byte-identical output (no git diff).
   uv run python tutorials/generate_tutorial.py
   git diff tutorials/
   ```
   Pre-commit hook also runs this — committed `.py`/`.ipynb` never drift.

5. **Keep training cheap by default**. Tutorials should smoke in a single
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
@code
def _define_model():
    class MyTinyModel(ModelConfiguration):
        model_name = "HuggingFaceTB/SmolLM2-135M"
        # For SLIME, also set architecture = ModelArchitecture(...)

        def download_model(self):
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=self.model_name)
```

Better: inherit from `HFModelConfiguration` and skip the `download_model`
override (the one-line snapshot_download is the inherited default). See
`tutorials/tutorial_generator/sft/ms_swift_custom_hf.py` for a full example.

## Validation

Always follow the tiered policy in
[agent-example-validation.md](agent-example-validation.md):

- **Tier 0 (local compile)** — `uv run python -m compileall modal_training_gym/`.
- **Tier 1 (cheap drift checks)** — regenerate tutorials (byte-determinism
  check) + local instantiation smoke across the affected frameworks. No GPU.
- **Tier 2 (scheduled smoke)** — one remote `modal run --detach` that
  reaches ≥1 training step, then kill the detached app.
- **Tier 3 (full example validation)** — canonical multi-node runs.
  Scheduled, not per-PR gating.

Per-change default: Tier 0 + Tier 1, plus Tier 2 for the new/modified
tutorial only. Don't expand to all tutorials on a single change.

`tests/test_model_configuration.py` is the current model-API regression
test. Run with `uv run python tests/test_model_configuration.py`.

## Gotchas

- **Python version pin**. The repo pins 3.12 (see `pyproject.toml` +
  `CLAUDE.md`). Modal's `serialized=True` functions require the remote
  image's Python to match the local one. If a framework image has Python
  3.11 (e.g. some ModelScope images), app build fails with `InvalidError`.
- **Framework image switches**. To override a framework's default image,
  set `image=` on `<F>FrameworkConfig` in your tutorial. The launcher's
  `pip_install` chain reinstalls the framework (e.g. ms-swift) fresh, so
  switching the base is usually enough. Check whether transitive deps
  (megatron-core, pillow, tokenizers for transformers) are in the new
  image; the ModelScope image shipped many, bare CUDA/NGC images don't.
- **cloudpickle caller_module**. `SlimeConfig.build_app()` (and the other
  wrapper `build_app` methods) now all delegate to the launcher, meaning
  `inspect.stack()[1]` inside `build_<f>_app` is the config wrapper, not
  the tutorial. Launchers use `resolve_caller_module()` to walk past
  `modal_training_gym.*` frames. Never use raw `inspect.stack()[1]` here.
- **Secrets required for most remote runs**: `huggingface-secret` (with
  `HF_TOKEN`) and `wandb-secret` (with `WANDB_API_KEY`) must exist as Modal
  Secrets in your environment.
- **Do not edit generated tutorials**. The pre-commit hook rewrites them.
- **Do not add framework-specific quirks to `<F>Config`** that only matter
  for one model. Put those in the model's `download_model` override and
  make the tool script live in `modal_training_gym/tools/`.

## Common file references

- Adding/modifying a model → `modal_training_gym/common/models/`.
- Adding/modifying a framework → `modal_training_gym/frameworks/<name>/`.
- Cross-framework scripts → `modal_training_gym/tools/`.
- Cross-framework helpers → `modal_training_gym/common/framework.py`.
- Tutorial sources → `tutorials/tutorial_generator/<name>.py`.
- Tutorial regeneration → `uv run python tutorials/generate_tutorial.py`.
- Tests → `tests/test_*.py`, run via `uv run python tests/<file>.py`.
