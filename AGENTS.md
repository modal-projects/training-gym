# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

`modal-training-gym` is a pip-installable Python package that provides framework-aware launchers for distributed training on Modal's multi-node GPU clusters. The current entrypoint is `TrainConfig` + a recipe (`SlimeRecipe` / `MilesConfig`), then `.train()` / `.launch()` — the package handles image construction, cluster topology, Ray/NCCL bring-up, volume mounts, and checkpointing.

## Commands

```bash
# Setup
uv sync                              # install deps (Python 3.12 required)
uv run pre-commit install            # register local hooks

# Lint (ruff — tutorials/ is excluded via pyproject.toml)
uv run ruff check modal_training_gym/
uv run ruff format --check modal_training_gym/

# Type check
uv run pyright modal_training_gym/    # if pyright is available

# Compile check (no GPU needed)
uv run -m compileall modal_training_gym/ tutorials/

# Docs (Astro/Starlight site at docs-next/)
uv run scripts/generate_all.py --skip-build   # regen models table, API reference, docs pages
cd docs-next && npm ci && npm run dev                 # local dev server
uv run scripts/generate_all.py                 # full regen + build

# Models table (generated from the recipe registries)
uv run scripts/generate_models_table.py         # regenerate
uv run scripts/generate_models_table.py --check # CI freshness check

# Deploy
# IMPORTANT: These commands are only for development of the gym itself.
# Consumers of the gym should use `training-gym setup` instead.
# Features such as requiring proxy authentication only work with the CLI
# and will stop working if the dashboard is deployed with `modal deploy`.
uv run modal deploy docs-next/docs_next_app.py        # docs site → gym.modal.dev
uv run modal deploy dashboards/app.py                  # observability dashboard

# Validate model configs / map a diff to affected tutorials
uv run scripts/validate_model_configs.py list                         # models + frameworks
uv run scripts/validate_model_configs.py list --names-only --pr-only # PR matrix names
uv run scripts/validate_model_configs.py check -m qwen3-4b
# miles models go through the same script; the registry picks the framework
uv run scripts/validate_model_configs.py check -m Qwen3.5-4B-Miles
git diff | uv run python -m scripts.diff_impact
```

## Architecture

### TrainConfig + recipe

`TrainConfig` composes `dataset`, `model`, and a recipe (`SlimeRecipe` or `MilesConfig`). Call `.train()` or `.launch()` — there is no public `build_app()`. Recipes carry Modal infra + framework CLI flags (`extra="forbid"`).

### Train pipeline

All frameworks expose a single `train()` entry point. Calling `train()` handles model download, dataset preparation, and training automatically — if the model isn't cached or the dataset isn't materialized, `train()` runs those steps first.

### Volume layout

Every framework mounts three Modal Volumes:
- `/root/.cache/huggingface` — shared HF model cache (read-mostly)
- `/data` — training data (framework-specific, per-app)
- `/checkpoints` — training outputs (per-app, persists across runs)

### Model presets

Known-model presets live under `train_recipes/` (e.g. `Qwen3_4b_Recipe`); `TrainConfig.merge_model_recipe` (bool, default `True`) merges them onto unset recipe fields.

### Model validation

One registry, one script, one workflow, across every framework.

`common/models/validation.py` holds `VALIDATION_CONFIGS`: each entry maps a model name to its `ModelConfig`, the framework whose `get_base_recipe` trains it, and `run_on_pr`. The framework has to be declared — `SlimeRecipe.get_base_recipe` returns a recipe for any model it's asked about, so the recipe classes can't answer "is this model mine?".

`scripts/validate_model_configs.py` owns everything framework-agnostic (CLI, result JSON, markdown summary, PR comment, and the `check` flags). `scripts/validation_backends/<framework>.py` owns the only two things that differ: which recipe trains the model and which dataset it trains on, returned as a pair from one `build_*_validation` function. Recipes are used as `get_base_recipe` returns them, image included — the image a miles model trains on is declared once, in `MilesRecipe`, and validating a candidate image means bumping it on a branch and dispatching, not passing a flag. Adding a framework is one module here plus registry entries.

`run_on_pr=False` marks a model too expensive to fan out on a PR: it remains runnable by name from the CLI or `workflow_dispatch`, but `diff_impact.py` never puts it in a PR matrix. `list` prints the whole registry with each model's framework so dispatch-only models are discoverable; `--names-only --pr-only` emits the name-only PR matrix used by the workflow's blank-dispatch branch. `tests/test_model_validation_registry.py` enforces both. `diff_impact.py` also scopes re-validation per framework, so a miles-only change doesn't re-run the slime set.

### Cloudpickle caller resolution

Launchers use `resolve_caller_module()` (in `common/framework.py`) to find the user's tutorial module by walking the stack past `modal_training_gym.*` frames. This enables cloudpickle to serialize inline `DatasetConfig`/`ModelConfig` subclasses by value to remote containers.

### TrainResult persistence

`TrainResult` is a dataclass written to the metadata volume (`MetadataStore.TRAIN_RESULTS`, keyed by `training_run_id`). Created by each framework's `train()` on rank 0. Loaded by eval scripts via `TrainResult.load(training_run_id)`. The `.model` property reconstructs a `ModelConfig` pointing at the checkpoint for serving.

### Tutorial system

Tutorials are flat runnable scripts at `tutorials/*.py`. Each starts with comment frontmatter whose only field is `order`. `docs-next/src/lib/tutorial-docs-loader.ts` discovers the corpus, validates contiguous order values, and renders Markdown comments plus Python code into docs pages.

### API reference generation

`scripts/api_reference_manifest.py` contains a curated list of public classes. `scripts/generate_api_reference.py` introspects each class (fields, types, defaults, methods) and generates Starlight markdown pages. Run via `scripts/generate_all.py`.

### Dashboard

`dashboards/app.py` is a Modal app with a Svelte frontend (built at image-build time). Training runs and evals write metadata to a shared Modal Volume (`training-gym-metadata`) via `modal_training_gym.utils.metadata`. The ASGI endpoint serves the pre-built SPA + JSON APIs (`/api/runs`, `/api/train-results`, `/api/evals`) that read summary JSON from the volume on demand.

## Working rules

- Use `uv` for all Python operations. Never install packages at the system level.
- Tutorial sources are the flat `tutorials/*.py` files. Keep `order` values contiguous from zero.
- Never hand-edit the Models table in README.md. It is generated from `__all__` of each `train_recipes/*_recipe/__init__.py`; add the recipe and matching `ModelConfig` export, then rerun `scripts/generate_models_table.py`.
- Ruff excludes `tutorials/**`.
- Python 3.12 is pinned. Modal's `serialized=True` requires local ↔ remote Python version match.
- Modal Secrets `huggingface-secret` (HF_TOKEN) and `wandb-secret` (WANDB_API_KEY) are optional: HF auth is only needed for gated/rate-limited Hub access, and `wandb-secret` only when a `WandbConfig` is passed.
- Custom SGLang and vLLM deployments (`CustomDeployment.launch()`) are public by default (`unauthenticated=True`). Pass `unauthenticated=False` to require Modal proxy auth (export `MODAL_KEY` (`wk-…`) / `MODAL_SECRET` (`ws-…`) in the launching shell, or eval/`generate`/teacher calls return HTTP 401). For calls from remote workers (custom rm/reward fns) to authenticated endpoints, also forward the pair into the worker via a `modal.Secret` — the driver shell env doesn't reach them.
- Every framework's Modal app is tagged with `_modal_framework`, `_modal_job_type=training`, and metric provider/project/group for dashboard auto-discovery (see `common/__init__.py: COMMON_TRAINING_GYM_TAGS`).

## Agent skills

- Before acting, inspect the descriptions in `skills/*/SKILL.md` and read every
  skill that matches the request. Do not assume the explicitly named skills
  below are the only available skills.
- For training lifecycle work, read `skills/agent-driven-training/SKILL.md`
  before acting. This includes launching, monitoring, inspecting, diagnosing,
  continuing, or promoting a Training Gym run.
- For raw Modal infrastructure work, read
  `skills/modal-infrastructure/SKILL.md` before acting. Use it for apps,
  containers, volumes, scheduling, image builds, caches, and endpoint
  authentication.
- For model support work, read `skills/model-support/SKILL.md` before acting. Use it when adding, debugging, validating, or productionizing new model support, especially Slime recipes and model configs.
