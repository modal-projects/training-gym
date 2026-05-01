---
title: Training Gym SDK
description: Reusable building blocks and runnable examples for distributed training on Modal.
editUrl: https://github.com/modal-projects/training-gym/edit/joy/add-good-eval-framework/README.md
---

:::caution[Important]
Modal's multi-node cluster training product is in early preview and not
generally accessible. Please [**contact us**](https://modal.com/slack)
for access.
:::


**[📖 Documentation](https://gym.modal.dev)** · **[Tutorials](https://gym.modal.dev/tutorials/)** · **[API Reference](https://gym.modal.dev/reference/)**

Distributed training on [Modal](https://modal.com) without hand-rolling a
launcher each time. Pick a training framework (slime,
MS-SWIFT), plug in a model + dataset config, and
`modal run` it — training-gym handles the image, the
cluster topology, the Ray/NCCL bring-up, volume mounts, and checkpointing.

Packaged as `modal-training-gym` — pip-install once, then import
framework-specific launchers from your own scripts or notebooks. Every
tutorial is a runnable `.py` file **and** a matching `.ipynb` with the same
steps narrated cell-by-cell — the notebook is the place to read the
walkthrough; this README is the map.

## Install

In a notebook or script:

```python
! pip install -q git+https://github.com/modal-projects/training-gym.git@main
```

Every generated tutorial notebook has this line as its first code cell.

## Quickstart

Run a tutorial — Qwen3-0.6B GRPO on GSM8K using slime:

```bash
uv run modal run --detach tutorials/rl/001_slime_intro/001_slime_intro.py::app.train
```

Or open the matching `.ipynb` in Jupyter / Modal Notebooks and run
cell-by-cell — each notebook is a self-contained walkthrough. See
[`tutorials/README.md`](/tutorials/) for the full catalog.

## Pick your framework

Each framework package exposes a config class (e.g. `SlimeConfig`,
`MsSwiftConfig`) that returns a `modal.App` via `.build_app()` with
`download`, `prepare_dataset`, and `train` functions. Shared
container objects (`DatasetConfig`, `Model`, `WandbConfig`) plug into
the framework config; each framework translates them into its own CLI
vocabulary.

| Framework | Good for | Tutorials |
|---|---|---|
| `slime` | GRPO / RL post-training — Ray + Megatron + SGLang | [001 Intro](https://github.com/modal-projects/training-gym/tree/joy/add-good-eval-framework/tutorials/rl/001_slime_intro), [002 Customizing](https://github.com/modal-projects/training-gym/tree/joy/add-good-eval-framework/tutorials/rl/002_customizing_your_slime_run), [003 LLM Judge](https://github.com/modal-projects/training-gym/tree/joy/add-good-eval-framework/tutorials/rl/003_slime_with_llm_as_judge) |
| `ms_swift` | ms-swift Megatron SFT (single- and multi-node) | [001 ms-swift](https://github.com/modal-projects/training-gym/tree/joy/add-good-eval-framework/tutorials/sft/001_ms_swift) |

"Thin" launchers give you a cluster and a `torchrun` — bring your own
training script. "Opinionated" launchers wrap a specific upstream framework
and expect you to configure it via that framework's CLI/YAML vocabulary.

Source in `modal_training_gym/frameworks/`. Runnable examples in
[`tutorials/README.md`](/tutorials/).

## Documentation

Full docs are hosted at **[gym.modal.dev](https://gym.modal.dev)**:

- [Tutorials](https://gym.modal.dev/tutorials/) — step-by-step runnable examples
- [API Reference](https://gym.modal.dev/reference/) — every public class documented with types and defaults

External resources:

- [Multi-node training guide (Notion)](https://modal-com.notion.site/Multi-node-docs-1281e7f16949806f966adedfe8b2cb74?pvs=4)
- [Multi-GPU Training on Modal](https://modal.com/docs/guide/gpu#multi-gpu-training)
- [Using CUDA on Modal](https://modal.com/docs/guide/cuda)
- [GPU Metrics](https://modal.com/docs/guide/gpu-metrics)

## Observability Dashboard

Training Gym ships an observability dashboard that shows all your
training runs, their status, and W&B metrics (inline sparklines) —
all in one place.

Deploy your own instance:

```bash
uv run modal deploy dashboards/app.py
```

Modal will print the dashboard URL. The dashboard:

- **Auto-refreshes** every 5 minutes via a cron job that scans your Modal apps for training runs
- **Links to W&B** — each run shows a direct link to its Weights & Biases project page
- **Inline metrics** — expand any run to see training loss/reward sparklines pulled from W&B

The dashboard matches the docs site color scheme and serves a Svelte
frontend from a Modal web endpoint.

## License

[MIT](https://github.com/modal-projects/training-gym/blob/joy/add-good-eval-framework/LICENSE).

---

# Developer Guide

## Layout

```
modal_training_gym/        ← installable package
├── common/                ← cross-framework classes (datasets, models, wandb, Ray cluster helpers)
└── frameworks/            ← one package per training framework (see tutorials/ for the full list)

tutorials/                 ← runnable examples — one folder per tutorial
├── tutorial_generator/    ← source files; each produces a .py + .ipynb
└── generate_tutorial.py   ← AST-walks the sources, regenerates .py + .ipynb

dashboards/                ← observability dashboard (deploy with `modal deploy dashboards/app.py`)
docs-next/                 ← Starlight docs site (deploy with `modal deploy docs-next/docs_next_app.py`)
skills/                    ← agent skills for navigating this repo
```

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
versions to match, and the framework images we use (slime nightly, NeMo
25.11) all ship py312.

## Authoring a new tutorial

See [`tutorials/README.md`](/tutorials/#authoring-a-new-tutorial)
for the generator-source format and the per-tutorial `TUTORIAL_METADATA`
schema.

## Contributing a new framework

1. Create `modal_training_gym/frameworks/<name>/` with `config.py`,
   `launcher.py`, `__init__.py`.
2. `build_<name>_app(*, modal, config, name=None) -> modal.App` is the
   public entrypoint — same shape as the existing frameworks. See
   `modal_training_gym/frameworks/slime/launcher.py` for a complete
   reference.
3. Add a new `tutorials/tutorial_generator/<tutorial>.py` demonstrating it,
   and run the generator.
4. Container configs (`dataset`, `model`, `wandb`) are interpreted
   explicitly in each framework — don't try to share CLI vocabularies.
   The common config classes stay pure data.

## Agent guide

- [Running training jobs on Modal](https://github.com/modal-projects/training-gym/blob/joy/add-good-eval-framework/skills/agent-modal-training.md)
