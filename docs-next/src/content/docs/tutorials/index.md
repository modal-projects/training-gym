---
title: Tutorials
description: Runnable Modal training examples across intro, RL, SFT, and infrastructure-focused walkthroughs.
editUrl: https://github.com/modal-projects/training-gym/edit/main/tutorials/README.md
---

The catalog of available tutorials — with difficulty, framework, cluster
shape, and one-click **Launch** buttons — lives in the
[project README](/#tutorials). This page covers how to run the
`.py` companions from the CLI and how to author a new tutorial.

## Running from the CLI

Each tutorial also ships a plain `.py` companion you can `modal run` directly.
The sources are written notebook-first and narrated for Modal Notebooks, so
treat the `.ipynb` as the canonical walkthrough — the `.py` is the same
content stripped of narration.

## Authoring a new tutorial

Tutorials are generated from a Python source file under
[`tutorial_generator/`](https://github.com/modal-projects/training-gym/tree/joy/update-docs-to-add-no-need-for-gpu-messages-for-notebooks-and-add-focus-on-rl-in-docs/tutorials/tutorial_generator). The generator AST-walks each
source and emits `tutorials/<bucket>/<name>/<name>.py` + `tutorials/<bucket>/<name>/<name>.ipynb`.
**Edit the source, not the generated files** — the pre-commit hook
(`.pre-commit-config.yaml`) regenerates on commit, so hand-edits to the
`.py` / `.ipynb` will be overwritten.

Regenerate manually after editing a source file:

```bash
uv run python tutorials/generate_tutorial.py
```

### Cell decorators

Each top-level function in the source file produces one cell, in source order.
Function names and argument lists are ignored — only the decorator and the
body matter.

- `@markdown` — the function's docstring becomes one markdown cell
  (`.ipynb`) or `#` comments (`.py`).
- `@code` — the function's body (dedented) becomes one code cell.
- `@shell("…")` — the string argument is emitted verbatim as a code cell
  (supports `! pip install …` shell magic for notebook installs).
- `@py_only` / `@notebook_only` — restrict a cell to one output format;
  stacks on top of `@markdown` / `@code` / `@shell`.

### `TUTORIAL_METADATA`

Every source file declares a module-level `TUTORIAL_METADATA` dict that
drives the catalog table above. Fields:

| Key | Required | Purpose |
|---|---|---|
| `framework` | yes | Backtick-wrapped framework name (e.g., `` '`slime`' ``) or `'— (…)'` for standalone tutorials |
| `cluster_shape` | yes | Human-readable shape (`'4 × 8×H200'`) — must match what the config actually launches |
| `summary` | yes | One-line description of what the tutorial trains |
| `difficulty` | recommended | One of `'Beginner'`, `'Intermediate'`, `'Advanced'`. Falls back to `—` if omitted. See the column description at the top of this page for what each tier means. |
| `order` | yes | Integer controlling row order in the catalog; lower appears earlier |

### Writing style

Treat the notebook as the tutorial's home — the root README and this index
are maps, the `.ipynb` is the walkthrough. Aim for roughly:

- An intro cell naming *what* it trains, *why* someone would run it, and
  *where to watch progress* (W&B project/group, Modal dashboard).
- A markdown cell before each `@code` block that names the non-obvious
  choices (why these hyperparams, why this cluster shape, why this chat
  template). Don't repeat what the code itself makes obvious.
- Custom pieces (reward functions, dataset preprocessing, model wrappers)
  get their own explanation cell.
- A "Run it" section splitting CLI invocation (`@py_only`) from
  cell-by-cell interactive invocation (`@notebook_only`).
- Optional: a "Serve / evaluate / next step" tail, where relevant.

[`000_rl_basics`](https://github.com/modal-projects/training-gym/blob/joy/update-docs-to-add-no-need-for-gpu-messages-for-notebooks-and-add-focus-on-rl-in-docs/tutorials/rl/000_rl_basics/000_rl_basics.ipynb) is the current
reference example for tutorial narration depth.
