"""Docs-only tutorial source for `001_wandb_integration`.

Rendered on the docs site by scripts/generate_tutorial_pages.py. Lives in a
docs-only bucket (see `_DOCS_ONLY_BUCKETS` in tutorials/generate_tutorial.py),
so it never emits a runnable tutorials/tools/ .py/.ipynb pair and never
appears in the repo README catalog table.
"""

TUTORIAL_METADATA = {
    "framework": "—",
    "cluster_shape": "—",
    "summary": "Stream training curves to Weights & Biases — setup, run naming, sweeps, and dashboard deep-links",
    "difficulty": "Beginner",
    "order": 20,
    "docs_only": True,
    "api_classes": [
        "WandbConfig",
        "TrainConfig",
        "SlimeRecipe",
        "TrainingGroup",
    ],
}


from tutorial_generator import code, markdown, shell


@markdown
def _intro():
    """
    # Weights & Biases integration

    Every training recipe can stream its training curves — reward, KL,
    response lengths, learning rate, and the rest of the framework's metrics
    — to [Weights & Biases](https://wandb.ai). W&B logging is opt-in per run:
    pass a `WandbConfig` to the recipe to enable it, drop it to disable it.
    The [observability dashboard](/tutorials/tools/000_observability_dashboard/)
    records where each run logged and wires up an **Open in W&B** button on
    the run page, so the two views stay one click apart.
    """


@markdown
def _prereqs():
    """
    ## Prerequisites

    Training containers read the W&B API key from a Modal Secret named
    `wandb-secret` (with a `WANDB_API_KEY` entry). Create it once per
    workspace — either at [modal.com/secrets](https://modal.com/secrets) or
    from the CLI:
    """


@shell("modal secret create wandb-secret WANDB_API_KEY=<your-api-key>")
def _create_secret():
    pass


@markdown
def _prereqs_details():
    """
    The secret is attached to the training containers automatically whenever
    a recipe has W&B enabled. To read the key from a differently named
    secret, set `WandbConfig(modal_wandb_secret_name="my-secret")`.
    """


@markdown
def _enable():
    """
    ## Enable logging on a recipe

    Pass `wandb=WandbConfig(project="…")` to any training recipe:
    """


@code
def _enable_example():
    from modal_training_gym import Qwen3_5_4b_Recipe, TrainConfig, WandbConfig
    from modal_training_gym.common.models import Qwen3_5_4B

    training_run = TrainConfig(
        model=Qwen3_5_4B(),
        dataset=my_dataset,  # any DatasetConfig
        recipe=Qwen3_5_4b_Recipe(
            num_rollout=40,
            wandb=WandbConfig(
                project="my-rl-project",
                group="lr-sweep",  # optional: organize related runs
            ),
        ),
    )
    training_run.train()


@markdown
def _fields():
    """
    The `WandbConfig` fields you'll actually use:

    | Field | What it does |
    |---|---|
    | `project` | W&B project the run logs to. |
    | `entity` | W&B team/user. Optional — when omitted, the default entity for your API key is resolved automatically. |
    | `group` | W&B group tag. Use one group per experiment family (e.g. a sweep) so W&B's *Group* view overlays their curves. |
    | `disable_random_suffix` | On by default — keeps run names stable instead of letting W&B append a random suffix. |
    | `modal_wandb_secret_name` | Modal Secret to read `WANDB_API_KEY` from. Defaults to `"wandb-secret"`. |

    See the [`WandbConfig` reference](/reference/core/wandbconfig/) for the
    full list.
    """


@markdown
def _preflight():
    """
    ## Fail fast: the W&B preflight

    A misconfigured W&B key normally surfaces as a `CommError` minutes into a
    run — after the cluster is up and the model is loaded. To avoid burning
    GPU time on that, the launcher runs a **preflight check before any GPU
    work**: it logs in with your key, resolves the entity, and verifies it
    can write to the configured project by creating (and immediately
    deleting) a probe run.

    If the key is missing, expired, or can't write to the project, the run
    fails immediately with an actionable error telling you which secret to
    fix — or to drop `wandb=` if you didn't want logging at all.
    """


@markdown
def _run_mapping():
    """
    ## How runs map to W&B runs

    Each training run gets an id like `tender-ranch-2275c004a3bf`; its W&B
    run id is the first 8 characters of that id, so runs are easy to
    correlate across the dashboard, W&B, and your terminal:

    - First attempt → W&B run id `<first 8 chars of the run id>`.
    - If a run is **preempted and retried**, attempt *N* logs to
      `<first-8>-aN` with resume enabled, so each attempt's curves stay
      separate but linked.

    The dashboard records the entity, project, group, and run id for every
    attempt — the **W&B** button on a run's detail page (see the
    [dashboard guide](/tutorials/tools/000_observability_dashboard/)) jumps
    straight to the matching W&B run.
    """


@markdown
def _sweeps():
    """
    ## Sweeps

    When launching variants with `TrainingGroup`, give them a shared
    `group=` so W&B overlays their reward curves in its Group view. The
    sweep grid can also override W&B fields per variant via dotted paths
    (e.g. `"recipe.wandb.group"`), just like any other recipe field.
    """


@markdown
def _disable():
    """
    ## Disabling logging

    W&B is entirely opt-in: omit `wandb=` from the recipe and nothing is
    logged, no secret is required, and no preflight runs. Training metrics
    are still captured by the observability dashboard either way.
    """
