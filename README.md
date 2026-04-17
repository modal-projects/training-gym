# Training Gym

> [!IMPORTANT]
> Modal's multi-node cluster training product is in early preview and not
> generally accessible. Please [**contact us**](https://modal.com/slack)
> for access.

Reusable building blocks + runnable examples for distributed training on
[Modal](https://modal.com). Packaged as `modal-training-gym` — pip-install
once, then import framework-specific launchers from your own scripts or
notebooks.

## Layout

```
modal_training_gym/        ← installable package
├── common/                ← cross-framework classes (pure data)
│   ├── dataset.py         ← `DatasetConfig` base + `prepare()` hook
│   ├── wandb.py           ← `WandbConfig`
│   ├── ray_cluster.py     ← `ModalRayCluster` (Ray head + workers on Modal clusters)
│   └── models/            ← `BaseModelType` enum + `Model` + per-family specs
└── frameworks/            ← one package per training framework
    ├── slime/             ← SLIME RLVR (GRPO, SGLang + Megatron, via Ray)
    ├── ms_swift/          ← ms-swift Megatron (SFT / LoRA)
    ├── megatron/          ← NVIDIA NeMo `megatron.bridge`
    ├── verl/              ← verl (GRPO, Megatron + vLLM, via Ray)
    ├── torchrun/          ← generic `torchrun`-launched user script
    └── hf_accelerate/     ← generic `accelerate launch`-driven user script

tutorials/                 ← runnable examples — one folder per tutorial
├── tutorial_generator/    ← source files; each produces a .py + .ipynb
│   ├── slime_gsm8k.py
│   ├── ms_swift_glm_4_7_gsm8k.py
│   ├── megatron_glm_4_7_longmit128k.py
│   ├── verl_qwen3_32b_gsm8k.py
│   ├── starcoder_llama2_7b.py
│   └── nanogpt_owt.py
└── generate_tutorial.py   ← AST-walks the sources, regenerates .py + .ipynb

dashboards/                ← Grafana-style dashboards for monitoring runs
skills/                    ← agent skills for navigating this repo
```

## Quickstart

### Installing the package

In a notebook / script:

```python
! pip install -q git+https://github.com/modal-projects/training-gym.git@joy/initial-setup
```

Every generated tutorial notebook has this line as its first code cell.

### Running a tutorial

Pick a framework and run it directly with `modal run`. Example —
Qwen3-4B GRPO on GSM8K using SLIME:

```bash
uv run modal run tutorials/slime_gsm8k/slime_gsm8k.py::app.download_model
uv run modal run tutorials/slime_gsm8k/slime_gsm8k.py::app.prepare_dataset
uv run modal run --detach tutorials/slime_gsm8k/slime_gsm8k.py::app.train
```

Or open the matching `.ipynb` in Jupyter/Modal Notebooks and run cell-by-cell.

## Frameworks

Each framework package exposes a `build_<name>_app(modal=..., config=...)`
factory that returns a `modal.App` with download / prepare / train functions
ready to be invoked from `modal run` or `app.run()`. Common-container
objects (`DatasetConfig`, `Model`, `WandbConfig`) are plugged into the
framework config; each framework translates them into its own CLI
vocabulary (SLIME uses `--prompt-data`, ms-swift uses `--dataset`, verl
uses Hydra-style `key=value`, etc.).

| Package                                | Launch mechanism                     | Tutorial                                 |
|----------------------------------------|--------------------------------------|------------------------------------------|
| `modal_training_gym.frameworks.slime`  | Ray cluster + `JobSubmissionClient`  | `tutorials/slime_gsm8k/`                 |
| `…frameworks.ms_swift`                 | `torchrun megatron sft …`            | `tutorials/ms_swift_glm_4_7_gsm8k/`      |
| `…frameworks.megatron`                 | `torchrun train_script.py` (JSON cfg)| `tutorials/megatron_glm_4_7_longmit128k/`|
| `…frameworks.verl`                     | Ray + verl Hydra overrides           | `tutorials/verl_qwen3_32b_gsm8k/`        |
| `…frameworks.torchrun`                 | `torchrun <user script>`             | `tutorials/starcoder_llama2_7b/`         |
| `…frameworks.hf_accelerate`            | `accelerate launch <user script>`    | `tutorials/starcoder_llama2_7b/` (same)  |

`starcoder_llama2_7b` showcases both `torchrun` and `hf_accelerate` side by
side — same training script, two launchers. `nanogpt_owt` is a
non-framework tutorial that clones karpathy/nanoGPT at image-build time and
wraps it in Modal functions directly.

## Authoring a new tutorial

Tutorials are generated from a Python source file under
`tutorials/tutorial_generator/`. The generator AST-walks each source and
emits `tutorials/<name>/<name>.py` + `tutorials/<name>/<name>.ipynb`.

Decorators the generator understands:

- `@markdown` — function's docstring becomes a markdown cell (`.ipynb`) or
  `#` comments (`.py`).
- `@code` — function's body (dedented) becomes a code cell.
- `@shell("…")` — the string arg is emitted verbatim as a code cell
  (supports `!pip install …` shell magic).
- `@py_only` / `@notebook_only` — restrict a cell to one output format;
  stack on top of `@markdown` / `@code` / `@shell`.

Function names and argument lists are arbitrary — only decorator + body
matter. Cells appear in source order.

Regenerate after editing a source file:

```bash
uv run python tutorials/generate_tutorial.py
```

The pre-commit hook (`.pre-commit-config.yaml`) runs the generator
automatically so committed `.py` / `.ipynb` never drift from the source.

## Contributing a new framework

1. Create `modal_training_gym/frameworks/<name>/` with `config.py`,
   `launcher.py`, `__init__.py`.
2. `build_<name>_app(*, modal, config, name=None) -> modal.App` is the
   public entrypoint — same shape as the existing frameworks. See
   `modal_training_gym/frameworks/verl/launcher.py` for a complete
   reference.
3. Add a new `tutorials/tutorial_generator/<tutorial>.py` demonstrating it,
   and run the generator.
4. Container configs (`dataset`, `model`, `wandb`) are interpreted
   explicitly in each framework — don't try to share CLI vocabularies.
   The common config classes stay pure data.

## Dev setup

```bash
# editable install + pinned dev deps (pre-commit, ipykernel if you want)
uv sync

# register this venv as a Jupyter kernel (one-time, for notebook work)
uv run python -m ipykernel install --user --name=modal-training-gym

# install the pre-commit hook locally
uv run pre-commit install
```

Project Python is pinned to 3.12 (see `.python-version` / `pyproject.toml`);
every `@app.function(serialized=True)` requires the local ↔ remote Python
versions to match, and a couple of the framework images we use ship py312
by default.

## Documentation

- [Multi-node training guide (Notion)](https://modal-com.notion.site/Multi-node-docs-1281e7f16949806f966adedfe8b2cb74?pvs=4)
- [Multi-GPU Training on Modal](https://modal.com/docs/guide/gpu#multi-gpu-training)
- [Using CUDA on Modal](https://modal.com/docs/guide/cuda)
- [GPU Metrics](https://modal.com/docs/guide/gpu-metrics)
- [Agent guide for running training jobs on Modal](skills/agent-modal-training.md)

## License

[MIT](LICENSE).
