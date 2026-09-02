# Adding a new model config to miles

Read [../SKILL.md](../SKILL.md) first for the artifacts directory, the gotchas that apply to both frameworks, and the shared validation gates.

Always read the common gotchas.

### Phase 1: Discovery

First try looking for the existing model running on miles. Upstream ships inside the pinned image at `/root/miles`: `scripts/models/<model>.py` (the `MODEL_ARGS` a recipe sources — newer images ship these as `.py` modules exposing `model_args()`, older ones as `.sh`; check which your pinned image has, the two are reached by different recipe fields — see below), `scripts/run_<model>.py` (the validated cluster shape, parallelism and hyperparameters), `docs/models/<vendor>/<model>.md`, `miles_plugins/models/<model>/`, and `miles/backends/megatron_utils/megatron_to_hf/<model>.py` (the weight mapping selected by `model_name`). Probe the image to read them — `scripts/fetch_miles_patch_snapshots.py` is the working pattern. If you cannot find an existing model, find the model with the most similar architecture. Reference huggingface for model architecture.

**Check image/version compatibility FIRST — it is the most common blocker.** The gym pins the miles image per recipe (`docker_image` on `MilesRecipe`, overridden on the recipe subclass), so a bump changes only the recipe you edit. A model added to miles *after* that image was built will not run on it. Verify:
- **When support landed upstream** — the model plugin, the `megatron_to_hf` mapping, **and** the sglang inside the image all have to be new enough. A model with custom kernels needs an sglang that can serve them, not just the plugin.
- **That the tag runs on Modal** — the named tag an upstream model doc recommends may be arm64-only (GB300), which will not run on H100/H200. The dated `dev-*` nightlies are multi-arch; prefer one pushed shortly after the upstream merge.
- **That the tag still exists** — radixark prunes dated dev tags, so a pin that once worked can 404. Modal's cache still serves existing apps, so a stale pin is invisible until a cold pull in a fresh environment fails. Report one you find on a recipe you're not otherwise changing rather than bumping it silently.

Then, output your first artifact, which is a file called `model_setup.md`, containing:
- Is there this model or a model with the same architecture that is already validated on miles?
- **Does upstream support postdate the recipe's pinned `docker_image`?** If so, note the required image bump + sglang version.
- Is this model validated to be supported in megatron?
- Is this model validated to be supported in sglang?
- What is your plan for train configuration?
- How long do you expect each step in training take?
- How long do you expect each substep to take (e.g. rollout server initialization, weight sync, rollouts, etc)

**Scale/cost gate.** The models that land on miles are large by construction. Do not launch a multi-node run without explicit user sign-off on GPU budget and cluster availability; write the recipe + config first (Phases 1 & 4) and leave the live run to the operator if unconfirmed.

### Phase 2: Implementation

Output a miles config you believe will work, then follow the one-step proof in
[agent-driven-training](../../agent-driven-training/SKILL.md). Output the config
in `configs` directly and track progress in
`progress_log_[attempt_count].md`.

While tracking the progress, also make sure the timing lines up with your expectations in the `model_setup.md` artifact.

If this step does not work, go back to phase 1: what assumptions did you make in phase 1 that were incorrect and caused this? Output an artifact if it fails with `failure_analysis_[attempt_count].md`.

Record how long the step took, and how long each substep took. Make sure the model parser works and it is not generating gibberish.

### Phase 3: Validation

Follow the smoke-test loop in
[agent-driven-training](../../agent-driven-training/SKILL.md) with about 10
steps. If it fails, create a minimal reproduction and address the cause.

Record how long the step took, and how long each substep took. Make sure the model parser works and it is not generating gibberish.

### Phase 4: Productionize

Create a doc describing the miles config changes, and justify any patches you have made. If it's possible to not patch, do not patch.


# Common gotchas

`patch_files` (patch scripts applied at image build) and `local_miles` (a local checkout mounted over the image's copy, no rebuild) are the two patching levers. Gate model-specific patches on the model — the hardcoded 30s router bind timeout and the VL prompt preprocessing are both patched that way. Patching miles' own sources has a snapshot contract — `tests/testdata/miles/*.input|.output` plus `validate-miles-patch-snapshots.yml`, refreshed with `uv run modal run scripts/fetch_miles_patch_snapshots.py` — so changing `docker_image` can break that CI even with no patch change. Prove a patch applied by printing the patched source out of the image; "didn't apply" and "applied and didn't help" look identical in a training log.

`megatron_to_hf_mode` picks the checkpoint path: `"bridge"` uses megatron-bridge, `"raw"` turns on HF → torch_dist conversion, `""` disables export. Only the non-bridge path needs Megatron's torch_dist save patches.

Miles registers every sglang `ServerArgs` option under a `--sglang-` prefix, so any rollout-engine knob is reachable as an `sglang_*` field with no gym code.

## How the recipe maps to CLI flags (add flags without touching gym code)

`MilesRecipe` inherits `BaseTrainRecipe.cli_args`, which emits `--<field-name-with-dashes> <value>` for **every dataclass field** not listed in `_MILES_SKIP` (recipe.py). So the way to add an arbitrary miles/sglang flag is simply to **declare it as a field on your recipe subclass** — no edits to `recipe.py` or the launcher. The existing recipes do exactly this for their `sglang_*` and perf flags. Rules `cli_args` follows:
- `True` → bare flag (`--foo`); `False` / `None` / `""` → omitted entirely. So default an unwanted flag to `None`/`False`/`""`.
- `list` → `--foo a b c`.
- Fields in `YAML_CONFIG_FIELDS` (`eval_config`, `extra_config`, `sglang_config`) may be passed as a **dict** — `prepare_miles_config` materializes it to a YAML file at runtime and rewrites the value to the path. `JSON_CONFIG_FIELDS` (`train_env_vars`, `apply_chat_template_kwargs`, `multimodal_keys`) are passed as JSON. `extra_config` is the escape hatch: its keys become miles args and override same-named fields.
- Things in `_MILES_SKIP` (e.g. `miles_model_script`, `megatron_conversion_hf_checkpoint`, `environment`) are launcher instructions, not CLI flags — they won't appear in `cli_args` output. That is expected, not a bug.
- `${MODEL_ARGS[@]}` are emitted **before** the recipe's flags (`build_train_cmd`), so a recipe field overrides the same flag baked into the model script. That is how `custom_model_provider_path` swaps a provider without forking the script.

## `miles_model_script` / `miles_model_name` vs. `ModelArchitecture`

When a model's args aren't representable in `ModelArchitecture` (custom kernels, exotic MoE routing, a custom model provider), point the recipe at upstream's model script instead. **Which field depends on how the image ships it:**

- `.sh` script -> `miles_model_script = "scripts/models/<model>.sh"`. The launcher `source`s it and passes `${MODEL_ARGS[@]}`.
- `.py` module exposing `model_args()` -> `miles_model_name = "<model>"` (no path, no extension). The launcher runs `miles/utils/external_utils/model_args_utils.py <name>` and splices the printed line in as `${MODEL_ARGS[@]}`.

Newer images have converted `scripts/models/*` to `.py`, so `miles_model_script` with a `.sh` path silently refers to a file that no longer exists on a bumped image. Verify against the image you pin. Either way the upstream args are used verbatim. The model class then leaves `architecture = None`. Arch args are not re-derived for the conversion step, so the script is the single source of truth. Also set `model_name` when the megatron→HF weight mapping is selected by name rather than by config.

## Checkpoint conversion (torch_dist) constraints

`get_checkpoint_conversion_policy` (modal_helpers/utils.py) decides the HF→torch_dist conversion layout. Gotchas:
- It emits **TP and PP** always, and **EP/ETP only when `conversion_expert_model_parallel_size` / `conversion_expert_tensor_parallel_size` are set explicitly**. torch_dist is reshardable, so training parallelism can differ from conversion — converting at TP8/PP1/EP8 while training at TP4/PP8/EP4 reloads fine, which is why most models leave the conversion EP unset and convert at the implicit EP1. Set them only when the full expert set does not fit a rank at EP1 (Inkling-Small, 256 experts); `etp * ep * pp` must divide `tp * pp` or the policy raises. The `decoder-first/last-pipeline-num-layers` pair is dropped when the conversion layout is PP1. `CONVERT_KEEP_PP1=1` stops `convert_hf_to_torch_dist.py` auto-bumping PP toward the rank count and rewriting the decoder split.
- **A Modal Volume cannot absorb a large sharded write.** The writer dies in `inline_container.cc` with `unexpected pos` when `--save` points at the Volume at 42 layers (fine at 4 and 8). `convert_via_local_staging` writes to local disk and moves shards over afterwards; budget `convert_ephemeral_disk_mb` for the whole checkpoint plus the in-flight shard, and expect the Volume copy to dominate (~45 min of a ~58 min conversion at 550 GB).
- **A crashed conversion can register as a cache hit.** `.metadata` is written last, so it is what separates a finished save from a dead one; without that check partial weights feed training.
- **`no_save_optim` must be paired with `no_load_optim`**, or resuming a params-only checkpoint dies with `KeyError: 'optimizer'`. Saving params only is often mandatory at scale — params plus distributed-optimizer state runs to terabytes, and the Volume buffers all of it to container-local disk — at the cost of restarting Adam moments on resume.

## Shipped callables and hooks

Miles takes custom functions as import paths; the gym ships the callable by value and writes the resolved path, via `_HOOK_PATH_FLAGS`, `_HOOK_PATH_CONFIG_KEYS` and `_HOOK_WRAPPER_PATHS` in recipe.py. The wrappers live in `frameworks/miles/phase_reporting.py` and run phase reporting and dashboard capture before delegating to yours — so **setting a raw `--*-path` yourself replaces the wrapper and the run trains fine while reporting no substep times**, failing the Phase-2 dashboard gate. Pass the callable on the recipe field instead. Prefer `custom_reward_post_process_function` over a dotted path: a `__main__` function has no importable module name and miles' `import_module` fails inside the Ray actor. `capture_trace` + `trace_sample_limit` attach a per-sample generate/reward/tool-call timeline, useful when diagnosing gibberish.

## Large multi-node models

Everything here was learned the expensive way on Nemotron-3-Ultra-550B (16 x 8 H200);
see `.gym/new_models/Nemotron3_Ultra_550B_A55B/failure_analysis_*.md` for the evidence.

**Start by diffing your recipe's resolved settings against the closest existing
large recipe, not against upstream's launcher.** `Inkling_Small_Recipe` is the
reference for multi-node miles on Modal. Four of its settings are deviations from
both `MilesRecipe`'s defaults and upstream's scripts, each encoding a failure that
only appears at scale. A new large recipe that does not carry them will hit the
same four failures in sequence, one 128-GPU run at a time:

| Setting | Default that breaks | What breaks |
|---|---|---|
| `offload_train_target="disk"` (+ `offload_train_disk_dir`) | miles defaults to `"cpu"` | The paused actor is backed up into host RAM *on top of* `--optimizer-cpu-offload`. Megatron actors segfault inside `offload_train`, Ray reports `ActorUnavailableError`, the gym retries silently. |
| `NCCL_NVLS_ENABLE="0"` | `MilesRecipe` defaults to `"1"` | NVLS is intra-node NVLink SHARP. A tensor-parallel group spanning nodes cannot use it, and `ncclCommInitRank` fails with `NCCL error: invalid usage` while building `_TP`, killing engine init. |
| `use_distributed_optimizer=True` | off in Megatron and in upstream's scripts | Without it every DP replica holds a full copy of the optimizer state. See the arithmetic below. |
| `RAY_memory_monitor_refresh_ms="0"` | Ray's monitor is on | Ray kills actors under host-memory pressure, which is the normal regime for a CPU-offloaded optimizer. |

`/tmp` is on the container's overlay filesystem on Modal, **not** a tmpfs, so
`/tmp/train_offload` is genuinely disk. Upstream's help text warns that a tmpfs
path silently defeats disk offload, so this is worth re-checking if the image
changes.

### Do the host-RAM arithmetic before launching

```
params_per_rank = total_params / (TP * PP)
optimizer_state = params_per_rank * 12 bytes     # fp32 master + 2 Adam moments
per_node        = optimizer_state * gpus_per_node
```

For 550 B at TP8/PP4 that is 17.2 B params/rank -> **1.50 TiB per 8-rank node**,
against a 2 TiB Modal H200 host — before the checkpoint staging
(`total_params * 2 / num_nodes`), the weights, and activations. It OOMs. With
`use_distributed_optimizer=True` the state shards across DP and the same number
is 0.38 TiB. The distributed optimizer is mathematically equivalent: it changes
where state lives, not the update.

### Multi-node rollout engines are their own shape

`rollout_num_gpus_per_engine > actor_num_gpus_per_node` means the engine spans
nodes (`nnodes>1` in sglang's `server_args`), which a single-node engine never
exercises. Two failures live only here:

- **SGLang's post-load barrier.** `UNBALANCED_MODEL_LOADING_TIMEOUT_S` is a
  hardcoded 480 s in `load_model_utils.py` with no flag and no env var. Only the
  node that *downloaded* the checkpoint reads it back from page cache (measured:
  40 s); every other node pulls it cold off the Modal Volume at ~1 GiB/s, so a
  1 TB checkpoint needs ~30 min. The first rank to finish then raises
  `ValueError: TP rank N could finish the model loading, but there are other ranks
  that didn't finish loading`, the engine tears down its TCPStore, and the ranks
  still loading emit 20 minutes of `Broken pipe` NCCL warnings — which is all you
  see unless you go looking for the original `ValueError`.
  `frameworks/miles/modal_helpers/patches/patch_sglang_load_barrier.py` makes the
  constant read `$MILES_LOAD_BARRIER_TIMEOUT_S`, defaulting to upstream's 480 so
  single-node models keep fast dead-rank detection; a recipe that needs longer
  sets it in `environment`.
- **Raise `rollout_health_check_first_wait` to match.** If the barrier allows
  3600 s but the health checker still allows 1800, you have only moved the
  failure.

### What a slice can and cannot prove

A pruned few-layer slice is the right first step — it is ~1/15 the cost and
catches real plumbing bugs — but be explicit about its blind spots. It **cannot**
exercise: multi-node rollout engines (its engine is `nnodes=1`), host-RAM limits,
the cold-Volume read path, or output quality and reward (see SKILL.md). Two of
the four failures above were invisible to a slice that had just run five clean
steps. Budget for the full-scale run finding new things; do not treat a green
slice as de-risking the launch.

### Iterating cheaply at multi-node scale

Read SKILL.md's "Iterate cheaply" section first; these are the miles-specific
mechanics, learned on Nemotron-3-Ultra where every observation behind a
transition cost ~25–40 min of bring-up.

**Warm save/offload lab.** miles ships `--save-trigger-sentinel`: if the file
exists at a rollout boundary, a checkpoint is saved and the file removed. So one
cluster held alive on cheap rollouts yields as many save→offload observations as
you want for a single bring-up:

```python
recipe = <Model>_Recipe(
    num_rollout=50,                    # cheap steps keep the cluster alive
    rollout_max_response_len=256,      # short decode, not fewer samples
    save_interval=999,                 # no periodic saves (None is rejected:
                                       # Megatron asserts it whenever --save is set)
    extra_config={"save_trigger_sentinel": "/tmp/save_now"},
)
```

```bash
uv run modal container list | grep <app-id>            # find the head container
uv run modal container exec --no-pty <ta-...> -- touch /tmp/save_now
```

Note the final rollout still saves regardless — `should_run_periodic_action`
returns True at `rollout_id == num_rollout - 1` — so keep `num_rollout` well
above where you'll stop, and stop the app from the CLI when done.

**Respect the minimum viable rollout shape.** Shrink cost via
`rollout_max_response_len` and step count, never below the shape where every
DP-attention group has work. On Nemotron-3-Ultra (4 engines × dp4 = 16 DP
groups), 4 prompts × 2 samples left most groups idle and sglang's scheduler
died with `AssertionError: extend-idle conversion expects an empty rank` on
every attempt — the lab never reached a single save. 16 prompts × 4 samples
@ 2048 is the floor that runs clean there; derive the equivalent for your
engine × dp layout before shrinking.

**Answer image/flag/artifact questions with a CPU probe, not a training run.**
The working patterns live in `.gym/new_models/Nemotron3_Ultra_550B_A55B/`:
`probe_miles_image.py` (inspect upstream support inside the pinned image),
`validate_cli_args.py` (parse the emitted args with miles' own argparse — this
is how `save_interval=None` handling was verified before a launch),
`configs/verify_built_image.py` (grep markers out of the exact image
`_build_miles_base_image` produces — a cached layer is otherwise
indistinguishable from a skipped patch). Checkpoint completeness is a
`modal volume ls` (256 `.distcp` + `.metadata` for a 128-rank save), not a
resave.

**Reuse warm state across runs.** The HF cache volume keeps the model download
(~12 min saved), and recently-written volumes read back faster; back-to-back
16-node runs came up in ~22 min vs ~38 cold. Launch the next experiment while
the analysis of the previous one is still in progress only if they answer
independent questions — otherwise you pay bring-up to reproduce a conclusion
the pending run would have handed you.

### Timing at scale is not extrapolable

Rollout generation on the full model came in ~20x slower than a slice-based
estimate (3.0 h vs a predicted 3-6 min for 256 x 8192 tokens), and decode
degrades superlinearly with sequence length — the same model at 2048 tokens ran
~4x faster *per token* than at 8192. Predict step time from a run at the real
shape, or say the number is unvalidated.

## LoRA

Ship a full-parameter and a `*_LoRA_Recipe` variant on a shared private base, matching whichever upstream gates in CI. They differ on more than the adapter: full-param pins `use_dynamic_batch_size=False` + `micro_batch_size=1` (dynamic packing exposes a PP-p2p × EP-all-to-all NCCL race on varlen shapes) and offloads the optimizer, while LoRA packs dynamically, keeps both runtimes resident (`no_offload_train` / `no_offload_rollout`) and syncs only the adapter — ~3 s vs ~50 s per rollout.
- **Upstream's LoRA learning rate is a trap.** Its 5e-6 default reads as "not learning" because zero-initialized B factors need hundreds of rollouts to show a delta; 2e-4 is the validated value. Suspect this before the reward function.
- Set `sglang_moe_runner_backend = "triton"` for MoE LoRA. sglang's `auto` picks marlin for INT4 checkpoints, whose LoRA MoE path hits an illegal memory access capturing decode CUDA graphs at every batch size.
- Keep `sglang_max_lora_rank == lora_rank` and `sglang_max_loras_per_batch = 1`; RL serves exactly the current policy's adapter.
- `experts_shared_outer_loras=True` shares one outer factor across routed experts, with expert-specific factors following EP.
- PEFT adapter export is not supported for every architecture, so "train LoRA, ship an adapter" may not be deliverable even when training works.

## Multimodal

Three layers have to line up: the **model** (a multimodal `custom_model_provider_path` overriding the text provider from the model script — the towers load from `--hf-checkpoint` and never enter the torch_dist checkpoint, so no re-conversion), the **rollout** (`sglang_enable_multimodal`, plus `use_rollout_routing_replay` indexed over the media-expanded sequence so replay stays aligned after each `<image>` expands), and the **data** (`multimodal_keys`, which `MultimodalDataset` emits and `MilesRecipe` forwards). Override `_fields` on the recipe to select the multimodal provider automatically when the attached dataset has `multimodal_keys`.

The failure modes are silent, so check the rendered prompt before believing a bad reward: a missing `<image>` placeholder means the image never reaches the model and every sample scores unparseable, images must be materialized files rather than URLs, `apply_chat_template` must be off, and a coordinate-scale mismatch floors every sample (a model answering on a 0–1000 grid against a reward expecting 0–1 fractions scores zero on correct answers).

## Known upstream conflicts

`--use-dynamic-batch-size` conflicts with `--qkv-format bshd`. `--use-rollout-routing-replay` reads `num_experts_per_tok`, which some configs don't expose. Activation recompute can hit `save_for_backward` on tuple-returning layers. Raise `rollout_health_check_first_wait` a lot when deepgemm compiles, or engines get killed during startup.

`WandbConfig` is effectively unsupported: the launcher passes an empty run id because the driver and the `RolloutManager` both `wandb.init()` under a shared `WANDB_RUN_ID` and the second dies with "run ID … is in use". Use the dashboard for step/substep timing.

## Registration checklist (Phase 4)

Wiring a new `<Model>` + `<Model>_Recipe` (usually plus `<Model>_LoRA_Recipe`) requires edits in all of:
1. `modal_training_gym/common/models/<model>.py` + export in `common/models/__init__.py` (import + `__all__`).
2. `modal_training_gym/train_recipes/miles_recipe/<model>.py` + export in `miles_recipe/__init__.py` (import + `__all__`) — export every variant.
3. Top-level `modal_training_gym/__init__.py`: add to `_EXPORTS` (lazy map) **and** `__all__`.
4. `MilesRecipe.get_base_recipe` (recipe.py): add the `model_name → Recipe()` branch. Without it the model gets no preset and every caller must pass a recipe explicitly.
5. `common/models/validation.py: VALIDATION_CONFIGS`: `_ValidationConfig("<Name>", <Model>, Framework.MILES)`, with `run_on_pr=False` if the shape is too expensive for every PR. Step 4 is a prerequisite — `build_miles_validation` raises if `get_base_recipe` returns `None` — and the dataset it picks is DAPO-Math-17k, so a non-math recipe needs that backend widened.

Verify with: `uv run -m compileall`, `uv run ruff check <files>`, `uv run pytest tests/test_miles_recipe_hooks.py tests/test_miles_runtime_env.py tests/test_miles_patches.py`, and a quick `python -c "from modal_training_gym import <Model>, <Model>_Recipe; r=<Model>_Recipe(); print(r.gpu_allocation.summary())"` — instantiating the recipe runs the GPU-allocation and parallelism validators, catching bad TP/PP/EP/node math before any Modal run.
