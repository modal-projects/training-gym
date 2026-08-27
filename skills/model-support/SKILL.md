---
name: model-support
description: Use when adding, debugging, validating, or productionizing support for a new base model or
  model-specific recipe in modal-training-gym, especially Slime and Miles recipes and model configs.
---

## Adding a new model config

When asked to add a new model example to SlimeRecipe or MilesRecipe, you should output artifacts in a temporary directory in `.gym/new_models/[model_name]/` folder. Once you are finished, add the finished config to the recipes folder.

Read the reference for the framework whose recipe you are adding — each carries its own four phases and common gotchas:

| Framework | Reference |
| --- | --- |
| Slime (`SlimeRecipe`) | [references/slime.md](references/slime.md) |
| Miles (`MilesRecipe`) | [references/miles.md](references/miles.md) |

### Gates for every validation run

Phase 2's single step and Phase 3's smoke test both have to show:
1. The model output is not gibberish and actually makes sense.
2. Step time **and substep times** on the dashboard — patches that break the gym's observability fail this gate even when training works.
3. A non-zero raw reward, and by Phase 3 an increasing one. GSM8K saturates for strong bases, so pick a harder task rather than reading a flat curve as breakage.

If reward doesn't climb, isolate which layer is at fault: data (`DatasetConfig` — check rendered prompts read correctly and labels match source rows), the run setup (reward function shape, too few steps / too small batches / too low lr), the recipe (sampling, stop tokens, response budget, masking, optimizer, parallelism), or framework plumbing (weight sync, stale weights, checkpoint conversion).

**Read a recorded rollout sample before theorising about a reward.** The samples are on the metadata volume (`training-rollouts/<run>__<rollout>.json`, with `prompt`, `response`, `parsed_response`, `score`). A zero reward has many candidate explanations and exactly one cheap way to discriminate between them.

**A pruned/sliced checkpoint cannot satisfy gates 1 and 3 at all.** A few-layer slice of a large model emits token salad, so every group scores 0, every advantage is 0, and `grad_norm` is 0 *by construction* — a GRPO step on it is a mathematical no-op. Slices validate plumbing and timing; output quality and reward need the real model. Upstream's own slice CI only asserts the metrics exist. Don't read a slice's flat reward as breakage, and don't claim the model works because the slice ran.

**One step is not a time series.** Cold-start step cost can be ~10x steady state (kernel autotune, compilation), and it is measured on *step 0* of every run. Comparing step 0 of two different runs and concluding "steady" is invalid — you need ≥3 steps in one run.

### Operating a long run

Multi-hour, multi-node runs fail in ways that are invisible if you only watch elapsed time.

**The gym retries the train function** (`retries=Retries(max_retries=10)` in `frameworks/miles/launcher.py`). From the outside a retry is indistinguishable from a long first attempt: the phase display simply returns to `Initializing`. Tells, cheapest first: `metadata.attempt_count` in `training-runs/<run>.json`, a `retrying after preemption or interruption (attempt N)` line, or a changed `SGLangEngine pid`. Check one of these before concluding a run is merely slow.

**`modal app logs <app>` returns only the last ~100 entries by default**, and the rolling window will have dropped the traceback you want by the time you look. Use `--tail 20000` (the maximum) to recover it. Do not depend on `-f` streaming for a long run — it dies silently and leaves you blind rather than erroring.

**The gym's own phase display localises a failure faster than the raw logs.** `Optimizer step -> Saving checkpoint -> Initializing` says the save died, without reading a single stack frame.

**Measure container memory with the cgroup, not by summing `VmRSS`.** Summing `/proc/*/status` across processes double-counts forked children's copy-on-write pages, so it spikes exactly when something forks — which looks like memory pressure causing the fork failure you are investigating. Read `/sys/fs/cgroup/memory/memory.usage_in_bytes` instead. (Measured on a real container: summed RSS said 1903 GB, the cgroup said 88 GiB.)

### Common gotchas (both frameworks)

Naming convention: For the model name in artifacts, it should be `_` separated by model family identifiers and replacing `.` for versioning (e.g. `Qwen3_4b`, `Qwen3_6_35b`).

`--max-tokens-per-gpu` is a flag for training, whereas `--rollout-max-response-len` is a flag for rollouts.

If it is a large MoE, you may need `--optimizer-cpu-offload`, `--use-precision-aware-optimizer`, and `--overlap-cpu-optimizer-d2h-h2d`

### Validate Model Configs

`.github/workflows/validate-models.yml` guards against performance regressions and is manually triggered — required before merging a new model, so ping someone who can dispatch it (escalate to Joy Liu). Register the model in `common/models/validation.py: VALIDATION_CONFIGS` with its `Framework`; `run_on_pr=False` means dispatch-only, not disabled. The recipe is used as `get_base_recipe` returns it, image included, so validating a candidate image means bumping it on a branch and dispatching.
