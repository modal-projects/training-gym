---
title: Training Gym SDK
description: Reusable building blocks and runnable examples for distributed training on Modal.
editUrl: https://github.com/modal-projects/training-gym/edit/main/README.md
---

**[📖 Documentation](https://gym.modal.dev)** · **[Tutorials](https://gym.modal.dev/tutorials/)** · **[API Reference](https://gym.modal.dev/reference/)**

Modal Training Gym is a Python SDK for distributed training on [Modal](https://modal.com) — so you don't have to hand-roll a launcher every time.
Pick a base model, a dataset, and a training framework; the gym handles cluster topology, Ray/NCCL bring-up, volume mounts, checkpointing, and serving for eval and rollouts.

## Quickstart

Install with pip:

```bash
pip install -q git+https://github.com/modal-projects/training-gym.git@main
```

Or pin it in `pyproject.toml` for uv:

```toml
training-gym = { git = "https://github.com/modal-projects/training-gym.git", branch = "main" }
```

Then import the building blocks from your own script:

```python
from modal_training_gym import TrainConfig
```

### Learning the API

The fastest path through the API is the [tutorials](https://github.com/modal-projects/training-gym/blob/main/tutorials). Each one
ships as a runnable `.py` **and** a paired `.ipynb` narrated cell-by-cell —
the notebook is the canonical walkthrough.

## Observability dashboard

Training Gym ships a dashboard that aggregates training runs, deployments,
and eval results in one place. Deploy your own copy:

```bash
modal deploy dashboards/app.py
```

Modal prints a URL where you can watch jobs in progress.

![Gym Observability Dashboard](https://raw.githubusercontent.com/modal-projects/training-gym/main/assets/observability_dashboard.png)

## Multi-node access

:::caution[Important]
Single-node training is open to everyone. Multi-node clusters — required
for larger models — are still in Beta.
[**Contact us on Slack**](https://modal.com/slack) for access.
:::


## Documentation

Full docs are hosted at **[gym.modal.dev](https://gym.modal.dev)**:

- [Tutorials](https://gym.modal.dev/tutorials/) — step-by-step runnable examples
- [API Reference](https://gym.modal.dev/reference/) — every public class documented with types and defaults

Modal platform references:

- [Using CUDA on Modal](https://modal.com/docs/guide/cuda)
- [GPU Metrics](https://modal.com/docs/guide/gpu-metrics)
- [Multi-node clusters (Beta)](https://modal.com/docs/guide/multi-node-training)
- [Multi-GPU training on Modal](https://modal.com/docs/guide/gpu#multi-gpu-training)

## License

[MIT](https://github.com/modal-projects/training-gym/blob/main/LICENSE).

---

# Developer Guide

## Layout

```
modal_training_gym/        ← installable package
├── common/                ← shared classes (datasets, models, eval, deployment, Ray helpers)
├── deploy_recipes/        ← serving presets for engines like SGLang and vLLM
├── frameworks/            ← launcher implementations that build Modal apps
└── train_recipes/         ← training presets such as SlimeRecipe

tutorials/                 ← runnable examples — one folder per tutorial
├── tutorial_generator/    ← source files; each produces a .py + .ipynb
└── generate_tutorial.py   ← AST-walks the sources, regenerates .py + .ipynb

dashboards/                ← observability dashboard (deploy with `modal deploy dashboards/app.py`)
docs-next/                 ← Starlight docs site (deploy with `modal deploy docs-next/docs_next_app.py`)
skills/                    ← agent skills for navigating this repo
```

## Dev setup

```bash
# editable install + pinned dev deps (pre-commit, etc.)
uv sync

# optional: register this venv as a Jupyter kernel for notebook work
uv run python -m ipykernel install --user --name=modal-training-gym

# install the pre-commit hook locally
uv run pre-commit install
```

Python is pinned to 3.12 (see `.python-version` and `pyproject.toml`). Modal's
`@app.function(serialized=True)` requires the local and remote Python versions
to match, and the framework images we ship (slime nightly, NeMo 25.11) are all
py312.

## Authoring a new tutorial

See [`tutorials/README.md`](/tutorials/#authoring-a-new-tutorial)
for the generator-source format and the per-tutorial `TUTORIAL_METADATA`
schema.

## Contributing a new recipe

1. Add a train recipe under `modal_training_gym/train_recipes/`, or a deploy
   recipe under `modal_training_gym/deploy_recipes/`.
2. If the recipe needs new runtime behavior, wire it into the relevant launcher
   or serving builder under `modal_training_gym/frameworks/` or
   `modal_training_gym/deploy_recipes/*/serve_*.py`.
3. Add or update a source tutorial under
   `tutorials/tutorial_generator/<bucket>/` and run the generator.
4. Keep shared container objects (`dataset`, `model`, `wandb`, `eval`)
   framework-agnostic — recipe layers do the translation into engine-specific
   flags.

## Agent guide

Working on this repo with an AI coding agent? See
[`skills/agent-modal-training.md`](https://github.com/modal-projects/training-gym/blob/main/skills/agent-modal-training.md) for guidance
on running training jobs on Modal.
