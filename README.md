# Training Gym

**[📖 Documentation](https://gym.modal.dev)** · **[Tutorials](https://gym.modal.dev/tutorials/)** · **[API Reference](https://gym.modal.dev/reference/)**

Modal Training Gym provides building blocks for you to compose distributed training runs on [Modal](https://modal.com)
without handrolling a training launcher each time.

Just pick a dataset, base model, and
training framework of choice, and let training-gym handle the
compute infrastructure, cluster topology, the Ray/NCCL bring-up, volume mounts,
checkpointing, and serving deployment for training/eval/rollouts.

## Usage

Packaged as `modal-training-gym` — pip-install once, then import
framework-specific launchers from your own scripts.

To teach you how to use the gym, we created [tutorials](./tutorials/) for you.

Every tutorial is a runnable `.py` file **and** a matching `.ipynb` with the same
steps narrated cell-by-cell — the notebook is the place to read the
walkthrough.

### Installation

### Package Install

```python
! pip install -q git+https://github.com/modal-projects/training-gym.git@main
```

Every generated tutorial notebook has this line as its first code cell.


Then, in a script, you can import `modal-training-gym` as follows to use our building blocks:
```python
from modal_training_gym import TrainConfig
```

### Observability Dashboard

Training Gym ships an observability dashboard that tracks training
runs, deployments, and eval results in one place.

We recommend you deploy your own instance of our observability dashboard by running.
```bash
modal deploy dashboards/app.py
```
It will print out the corresponding URL that you can visit to look at your training run status

![Gym Observability Dashboard](./assets/observability_dashboard.png)

### Note for multi-node

Currently, all users can use the gym for single-node training.


> [!IMPORTANT]
> Modal’s training SDK requires multi-node clusters for larger models. Please [**contact us**](https://modal.com/slack)
> for access.


## Documentation

Full docs are hosted at **[gym.modal.dev](https://gym.modal.dev)**:

- [Tutorials](https://gym.modal.dev/tutorials/) — step-by-step runnable examples
- [API Reference](https://gym.modal.dev/reference/) — every public class documented with types and defaults

External resources:
- [Using CUDA on Modal](https://modal.com/docs/guide/cuda)
- [GPU Metrics](https://modal.com/docs/guide/gpu-metrics)
- [Multi-node clusters (Beta)](https://modal.com/docs/guide/multi-node-training)
- [Multi-GPU Training on Modal](https://modal.com/docs/guide/gpu#multi-gpu-training)

## License

[MIT](LICENSE).

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

See [`tutorials/README.md`](tutorials/README.md#authoring-a-new-tutorial)
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

- [Running training jobs on Modal](skills/agent-modal-training.md)
