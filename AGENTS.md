# CLAUDE.md

# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.



## What this repo is

`modal-training-gym` is a pip-installable Python package that provides framework-aware launchers for distributed training on Modal's multi-node GPU clusters. Users import framework configs, attach a model + dataset, and `modal run` — the package handles image construction, cluster topology, Ray/NCCL bring-up, volume mounts, and checkpointing.

## Commands

```bash
# Setup
uv sync                              # install deps (Python 3.12 required)
uv run pre-commit install            # register tutorial-regen hook

# Lint (ruff — tutorials/ is excluded via pyproject.toml)
uv run ruff check modal_training_gym/
uv run ruff format --check modal_training_gym/

# Type check
uv run pyright modal_training_gym/    # if pyright is available

# Compile check (no GPU needed)
uv run python -m compileall modal_training_gym/

# Tutorials — NEVER edit generated files directly
uv run python tutorials/generate_tutorial.py              # regenerate all .py + .ipynb
uv run python tutorials/generate_tutorial.py path/to/src  # regenerate one

# Docs (Astro/Starlight site at docs-next/)
uv run python scripts/generate_all.py --skip-build   # regen API reference + tutorial pages
cd docs-next && npm ci && npm run dev                 # local dev server
uv run python scripts/generate_all.py                 # full regen + build

# Deploy
uv run modal deploy docs-next/docs_next_app.py        # docs site → gym.modal.dev
uv run modal deploy dashboards/app.py                  # observability dashboard

# Validate tutorials (runs on Modal — costs GPU time)
uv run python scripts/validate_tutorials.py --list           # show discovered targets
uv run python scripts/validate_tutorials.py --preflight-only # local checks only
uv run python scripts/validate_tutorials.py --only slime_gsm8k  # single tutorial
```

## Architecture

### Two-class framework pattern

Every framework exposes two config classes:

- **`<F>FrameworkConfig`** — Modal infra (gpu, image, n_nodes) + framework CLI flags. Pydantic with `extra="forbid"`.
- **`<F>Config`** — Composes `dataset: DatasetConfig`, `model: ModelConfig`, `wandb: WandbConfig`, and `framework_config`. Exposes `build_app()`.

```
SlimeConfig.build_app()
  → build_slime_app(slime=self)  [in launcher.py]
    → returns modal.App with:
        app.download()          — ModelConfig.download()
        app.prepare_dataset()   — DatasetConfig.prepare()
        app.train()             — Ray cluster submit → TrainResult
```

SlimeConfig is a Pydantic dataclass following the two-class composition pattern.

### Train pipeline

All frameworks expose a single `train()` entry point. Calling `train()` handles model download, dataset preparation, and training automatically — if the model isn't cached or the dataset isn't materialized, `train()` runs those steps first.

### Volume layout

Every framework mounts three Modal Volumes:
- `/root/.cache/huggingface` — shared HF model cache (read-mostly)
- `/data` — training data (framework-specific, per-app)
- `/checkpoints` — training outputs (per-app, persists across runs)

### Model presets

Models can declare `training: ModelTrainingConfig` or framework-specific presets (e.g. `SlimePreset`) with tuned parallelism/GPU settings. Framework configs apply these as defaults to unset fields during `__post_init__`.

### Cloudpickle caller resolution

`build_app()` factories use `resolve_caller_module()` (in `common/framework.py`) to find the user's tutorial module by walking the stack past `modal_training_gym.*` frames. This enables cloudpickle to serialize inline `DatasetConfig`/`ModelConfig` subclasses by value to remote containers.

### TrainResult persistence

`TrainResult` is a dataclass written to a `modal.Dict` (keyed by `{app_name}-train-results`). Created by each framework's `train()` on rank 0. Loaded by eval scripts via `TrainResult.load(app_name)`. The `.model` property reconstructs a `ModelConfig` pointing at the checkpoint for serving.

### Tutorial system

Tutorials live in `tutorials/tutorial_generator/<bucket>/<name>.py` as decorator-annotated source files. The generator (`tutorials/generate_tutorial.py`) AST-walks each source and emits `tutorials/<bucket>/<name>/<name>.py` + `.ipynb`. The pre-commit hook auto-regenerates on commit.

Decorators: `@markdown` (docstring → md cell), `@code` (body → code cell), `@shell("...")` (verbatim cell), `@py_only` / `@notebook_only` (restrict output format).

Each source declares `TUTORIAL_METADATA` dict with `framework`, `cluster_shape`, `summary`, `difficulty`, `order`, `api_classes` — this drives the catalog table in the repo-root README.md and backlinks in API reference pages.

### API reference generation

`scripts/api_reference_manifest.py` contains a curated list of public classes. `scripts/generate_api_reference.py` introspects each class (fields, types, defaults, methods) and generates Starlight markdown pages. Run via `scripts/generate_all.py`.

### Dashboard

`dashboards/app.py` is a Modal app with a Svelte frontend (built at image-build time). Training runs, deployments, and evals write metadata to a shared Modal Volume (`training-gym-metadata`) via `modal_training_gym.utils.metadata`. The ASGI endpoint serves the pre-built SPA + JSON APIs (`/api/runs`, `/api/train-results`, `/api/evals`, `/api/deployments`) that read summary JSON from the volume on demand.

## Working rules

- Use `uv` for all Python operations. Never install packages at the system level.
- Never edit `tutorials/<name>/<name>.py` or `.ipynb` — they are generated. Edit `tutorials/tutorial_generator/<name>.py` and run the generator.
- Ruff excludes `tutorials/**` — generated tutorial code is not linted.
- Python 3.12 is pinned. Modal's `serialized=True` requires local ↔ remote Python version match.
- Modal Secrets `huggingface-secret` (HF_TOKEN) and `wandb-secret` (WANDB_API_KEY) are required for remote runs.
- Every framework's Modal app is tagged with `_modal_framework`, `_modal_job_type=training`, and W&B project/group for dashboard auto-discovery (see `common/__init__.py: COMMON_TRAINING_GYM_TAGS`).


