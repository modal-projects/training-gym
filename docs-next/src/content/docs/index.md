---
title: Training Gym SDK
description: Reusable building blocks and runnable examples for distributed training on Modal.
editUrl: https://github.com/modal-projects/training-gym/edit/joy/update-documentation/README.md
---

:::caution[Important]
Modal's multi-node cluster training product is in early preview and not
generally accessible. Please [**contact us**](https://modal.com/slack)
for access.
:::


**[📖 Documentation](https://gym.modal.dev)** · **[Tutorials](https://gym.modal.dev/tutorials/)** · **[API Reference](https://gym.modal.dev/reference/)**

Distributed training on [Modal](https://modal.com) without hand-rolling a
launcher each time. Compose a `TrainConfig` or `DeploymentConfig` from
shared model, dataset, and recipe objects, and training-gym handles the
image, the cluster topology, the Ray/NCCL bring-up, volume mounts,
checkpointing, and serving deployment.

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

The current end-to-end tutorial is notebook-first:

- [`tutorials/rl/000_rl_basics/000_rl_basics.ipynb`](https://github.com/modal-projects/training-gym/blob/joy/update-documentation/tutorials/rl/000_rl_basics/000_rl_basics.ipynb)
  walks through serving `Qwen3_4B`, evaluating it, training it with
  `SlimeRecipe`, and comparing the result.

The main public surfaces are:

- `TrainConfig` combines `DatasetConfig`, `ModelConfig`, and a training
  recipe such as `SlimeRecipe`.
- `DeploymentConfig` combines a `ModelConfig` with a serving recipe such
  as `SglangRecipe` or `VllmRecipe`.
- Built-in model presets: `Qwen3_0_6B`, `Qwen3_4B`, `Qwen3_30B`.

See [`tutorials/README.md`](/tutorials/) for the current catalog.

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

Training Gym ships an observability dashboard that tracks training
runs, deployments, and eval results in one place.

Deploy your own instance:

```bash
uv run modal deploy dashboards/app.py
```

Modal will print the dashboard URL. The dashboard:

- **Live data** — training runs, deployments, and evals are recorded to a shared metadata volume as they happen; the dashboard reads on demand
- **Three views** — Training, Deployments, and Evals pages with status tracking and Modal app links
- **Links to Modal** — each run links to its Modal app dashboard for logs and GPU metrics

The dashboard is a Svelte SPA served from a Modal ASGI endpoint.

## License

[MIT](https://github.com/modal-projects/training-gym/blob/joy/update-documentation/LICENSE).

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

## Contributing a new recipe

1. Add a train recipe under `modal_training_gym/train_recipes/` or a
   deploy recipe under `modal_training_gym/deploy_recipes/`.
2. If the recipe needs new runtime behavior, wire it into the relevant
   launcher or serving builder under `modal_training_gym/frameworks/` or
   `modal_training_gym/deploy_recipes/*/serve_*.py`.
3. Add or update a source tutorial under
   `tutorials/tutorial_generator/<bucket>/` and run the generators.
4. Keep shared container objects (`dataset`, `model`, `wandb`, `eval`)
   framework-agnostic; recipe layers should do the translation into
   engine-specific flags.

## Agent guide

- [Running training jobs on Modal](https://github.com/modal-projects/training-gym/blob/joy/update-documentation/skills/agent-modal-training.md)
