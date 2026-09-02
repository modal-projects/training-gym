# Porting `multiturn_slime` into main — agentic recipe slice

Running notes on the slices merged so far and what's still outstanding.
Rebased onto `2d63947c` on 2026-09-02.

## Where things stand

| Slice | PR | State |
|---|---|---|
| Slime git overlay / fork pinning | [#438](https://github.com/modal-projects/training-gym/pull/438) | merged |
| Trackio | [#454](https://github.com/modal-projects/training-gym/pull/454) | merged as `6ea030cd` |
| Qwen3.6-27B agentic recipe | — | this branch, `multiturn-slime-recipe` |

Four unrelated refactors landed on main underneath this work, all touching the
same files: `#445` removed recipe merging (`TrainConfig.merge_model_recipe` is
gone), `#465` moved shared topology defaults up into `SlimeRecipe`/`MilesRecipe`
and added the preset-override invariant test, `#480` reworked the reference
docs, and `#488` made class capitalization consistent (`Qwen3_6_27b_Recipe` →
`Qwen3_6_27B_Recipe`) and rewrote docstrings into an `Attributes:` style.

---

## 1. `Qwen3_6_27B_Recipe_Agentic`

48 GPUs (6×8×H200): 2 actor nodes disaggregated from 32 rollout GPUs, TP4 /
PP2 / CP2, Harbor agent rollouts against a pinned `modal-projects/slime` fork
at `ba324beb`. Trackio project defaults to `agentic-harbor`.

Three deviations from the version on `multiturn_slime`:

1. **`trackio=TrackioConfig(...)` → `metrics=TrackioConfig(project="agentic-harbor")`.**
   `multiturn_slime` carries a separate `trackio` recipe field, a slime source
   patch (`patch_trackio.py`) adding `--use-trackio`/`--trackio-project`, and a
   hand-rolled `/api/bulk_log` poster. Main took a different design: one
   `metrics: MetricConfig` field, `metric_cli_fields` emitting slime's existing
   `--use-wandb …` flags, and a `.pth` in site-packages that swaps in a fake
   `wandb` module backed by trackio. None of the `multiturn_slime` Trackio code
   was ported — only the intent.
2. **`dashboard_auth=True` dropped** — it depends on `common/auth_proxy.py`,
   which isn't on main. Deferred.
3. **Named `Qwen3_6_27B_Recipe_Agentic`**, following
   `Qwen3_6_35B_Recipe_Long_Context` — the variant suffix goes *after*
   `_Recipe`. Not cosmetic: `generate_models_table.py` derives a recipe's model
   key by stripping `_Recipe`, so `Qwen3_6_27b_Agentic_Recipe` reduced to
   `"qwen3_6_27b_agentic"`, matched no exported `ModelConfig`, and `SystemExit`ed
   a script that gates CI and pre-commit. The conventional name reduces to
   `"qwen3_6_27b"` and resolves on its own, so no `model_config_class` override
   is needed.

### `SlimeRecipe.data_volume_name`

New optional field (`None` → the launcher keeps deriving `<prefix>-data` from
the recipe class). Added to `_SLIME_SKIP` so it never leaks onto the slime
command line. The preset uses it to mount the existing `slime-data` volume at
`/data`.

### Fallout from `#465`

The preset-override invariant compares every preset against
`SlimeRecipe`/`MilesRecipe`. Two changes:

- Twelve field restatements removed from the preset
  (`tensor_model_parallel_size`, `pipeline_model_parallel_size`,
  `conversion_pipeline_model_parallel_size`, `decoder_last_pipeline_num_layers`,
  `n_samples_per_prompt`, `rollout_num_gpus_per_engine`,
  `actor_num_gpus_per_node`, `eval_interval`, …). Effective config is
  unchanged — the values are inherited now instead of repeated.
- The invariant baselines on `recipe_cls.__mro__[1]` rather than the framework
  class. This preset is the first to subclass *another preset*, where resetting
  a field back to the framework default is a real override. `rm_type=None` and
  `ref_load=""` are load-bearing: reward comes from the Harbor rollout, not a
  reward model, and there is no pre-converted reference checkpoint.

---

## 2. Trackio adapter fix — `common/trackio.py`

Found by actually running the recipe (see below), not by review.

`install_wandb_shim` publishes `wandb.run` as a **module global**, matching W&B,
where the run is process-global. trackio keeps its active run in
`context_vars.current_run`, a **`ContextVar`** — and a thread started after
`init()` begins with an empty context, so `current_run.get()` is `None` there.

That breaks an invariant callers rely on: *`wandb.run is not None` implies
`wandb.log()` works*. Slime's SGLang engine-metrics loop is a daemon thread
that guards exactly that way:

```python
if not metrics or wandb.run is None:
    continue
wandb.log(payload)          # raised: Call trackio.init() before trackio.log()
```

**Fix:** the shim's `log()` now goes through the run object it already holds
(`_RunProxy`) instead of the module-level `trackio.log()`, so any thread reaches
the same run. `trackio.log()` remains the fallback when no run is initialised,
preserving today's error there.

`finish()` has the same ContextVar dependency and would raise identically if
ever called off the init thread. It isn't today, and routing it through the run
would skip trackio's `current_run.set(None)` cleanup, so it is left alone.

**Why review didn't catch it:** the only caller that logs from a non-init thread
is `_engine_metrics_loop` in `slime/utils/wandb_utils.py`, and that file does
not exist in upstream `THUDM/slime` — it is added by the pinned fork. Everything
on main logs from the main thread, where the ContextVar is set, so the gap was
unreachable. `test_trackio_wandb_adapter_covers_the_framework_surface` exercises
the whole W&B surface but only from the test's own thread.

Regression test: `test_trackio_adapter_logs_from_a_thread_started_after_init`,
against a fake trackio whose run lives in a real `ContextVar`.

---

## 3. Verified by running it

Run `timid-smooth-e25b81b20798`, 1×8 H200 (the tutorial's smoke profile:
colocated, CP=1, so TP4×PP2×CP1 = 8 = actor world size), 2 rollouts over a
2-row subset with two eval subsets.

Confirmed: fork overlay and pin, cached torch_dist conversion reused, `/data`
mounted from `slime-data`, sglang engines up, agentic episodes running real
tool calls and being graded, `rollout 0` metrics, both eval subsets evaluated,
and `eval/train-2-smoke` / `eval/eval-2-smoke` arriving as distinct series.
Zero errors on the main logging path.

Not observed: no optimizer step completed — the run was stopped during rollout
1's eval. The same topology completed 2 full steps plus a checkpoint save on
the pre-port `multiturn_slime` version.

Launcher used: `agentic_smoke.py` (not in the repo; the `010_agentic_harbor`
tutorial it derives from is still unported).

---

## 4. Still on `multiturn_slime`

- `common/auth_proxy.py` + `dashboard_auth` on `SlimeRecipe` + the launcher's
  `AsyncExitStack` dashboard-forwarding rewrite + `tests/test_auth_proxy.py`
- `scripts/partition_harbor_dataset.py` + `tests/test_partition_harbor_dataset.py`
- the `010_agentic_harbor` tutorial
- `run_summary.trackio_url` — superseded by main's provider-agnostic
  `metric_*` summary fields

---

## Verification

```bash
uv run pytest tests/ -q                           # 919 passed, 1 skipped
uv run ruff check modal_training_gym/ tests/
uv run ruff format --check modal_training_gym/ tests/
uv run scripts/generate_models_table.py --check    # 17 models, up to date
```
