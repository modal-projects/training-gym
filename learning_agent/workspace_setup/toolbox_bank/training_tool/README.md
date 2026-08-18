# training_tool, the training packages

The packages pinned for your task's training methods, in
`toolbox/repos.yaml` (materialized by `clone_repos.py`):

<!-- if:sft,dpo -->
- `axolotl/`: SFT and DPO. Config-driven: LoRA, full fine-tuning, and
  preference training (`rl: dpo`; see its docs/rlhf.qmd). Read its own
  docs, compose its commands, run them on GPUs with
  `toolbox/gpu_tools/gpu_launcher.py`.
<!-- endif:sft,dpo -->
<!-- if:rl,opd -->
- `training_gym/`: RL (GRPO) and on-policy distillation, through the
  Training Gym SDK. The full guide is below; the playbooks in
  `training_gym/skills/` cover the whole lifecycle
  (`agent-driven-training` first).
<!-- endif:rl,opd -->
<!-- if:opd -->
- `self_distillation/`: on-policy SELF-distillation (the OPD variant
  with no separate teacher). The teacher
  is the same model with a context file in its window; the task model
  learns to answer without it. `main.py` is the entry; it saves only
  every 100 optimizer steps, so runs shorter than 100 steps save
  nothing. The data-transform variant (rewrite reference answers in the
  model's own words, then normal SFT) is described in
  `data_tool/self_distill/self_distill.md`.
<!-- endif:opd -->

Rules:

1. Pick the package for your method (the method list in AGENTS.md), read
   its own docs and recipes, compose its commands.
2. Run GPU work through `toolbox/gpu_tools/gpu_launcher.py` (image, volumes,
   H200s), or write your own Modal app; `toolbox/gpu_tools/` teaches how.
   The gym launches its own GPU jobs.
3. Every trained checkpoint ends at `/out/models/<tag>/merged`, the
   serving and scoring contract. Tags share a flat namespace: prefix them
   with something run-unique.
4. Only fine-tune the task model (rule 5 in AGENTS.md).

<!-- if:rl,opd -->
## The Training Gym: RL (GRPO)

`gpu_launcher.py` runs any command on a GPU with zero instrumentation.
The gym is the other end of that trade: it runs GRPO jobs, brings up the
cluster, the SGLang rollout engines, and the checkpointing itself, and
records what happened while it did. You drive the SDK directly;
`eval_tool/gym_eval.py` puts your dev set, judged by luna, on the same
dashboard. Before your first run, read the
`training_gym/skills/agent-driven-training` playbook.

### Install first (about a minute)

The `training-gym` CLI does not exist until you install the SDK. If you
see `training-gym: command not found`, you skipped this step; install,
don't fall back to browsing the source:

```bash
pip install -e toolbox/training_tool/training_gym   # the pinned SDK clone; Python 3.12
training-gym setup                      # deploys the dashboard into your Modal env
training-gym open                       # opens it
```

It uses the same Modal credentials `gpu_launcher.py` does. Everything lands in
`MODAL_ENVIRONMENT` (`lab-dev` for your run), so the
dashboard is shared with the team just like the observatory.

### Train

The data contract is unchanged: prompts shaped as in
`data_tool/rl/README.md` and an `async def rm(args, sample, **kwargs) -> float`
reward. The SDK's own README and `tutorials/agent/` (inside the clone) are the
full guide; the skeleton is:

Your task model is ALREADY SUPPORTED: the SDK ships a ready-made model
config and recipe for Qwen3.5-9B, import them, do not write your own:

```python
from modal_training_gym import TrainConfig, Qwen3_5_9B, Qwen3_5_9b_Recipe

result = TrainConfig(
    model=Qwen3_5_9B,   # the pinned task model, preconfigured
    dataset=...,        # a DatasetConfig over your prompts: {"messages": [...],
                        # "label": <what only the reward sees>}
    recipe=Qwen3_5_9b_Recipe(custom_rm_function=rm, gpu_type="H200",
                             num_rollout=64, n_samples_per_prompt=8),
).train()               # prints a training_run_id; `training-gym open` to watch
```

The examples and tutorials in the clone use other models (a Qwen3-4B
quickstart, a haiku toy reward), they show the shapes, not your setup.
The per-model recipes live in
`modal_training_gym/train_recipes/slime_recipe/` (qwen3_5_9b.py is
yours). The `skills/agent-driven-training` playbook walks the whole
lifecycle; read it before your first run.

Checkpoints land on the gym's own Modal volume
(`modal_training_gym.list_checkpoints(training_run_id)`). Copy the one you
keep to `lab-out:/out/models/$LEARNING_AGENT_RUN_ID/<tag>/merged`, the path
`submission/serve.py` and the operator harness expect. While iterating, skip the copy and pass
`--training-run-id` to `gym_eval.py` instead, that serves straight off the
gym's volume.

Multi-turn harnesses are supported through `custom_generate_function`: the
rollout is a whole episode, and you
control which tokens are trained via `sample.loss_mask`, set `0` on every
token you pasted in (tool output, environment feedback) or the task model learns
to imitate your search engine.

### Evaluate

```bash
python3 toolbox/eval_tool/gym_eval.py --dev task/dev.json \
    --harness harness.py:answer --label base
python3 toolbox/eval_tool/gym_eval.py --dev task/dev.json \
    --training-run-id <id> --harness harness.py:answer --label trained
```

Same judge as `rubric_eval.py` (`judge_client`, luna, n-vote majority per
claim) and the same seeded bootstrap CI, so the numbers are comparable with
the ones already in your log. The difference is where they land: two rows on
the Evals tab, each question drillable down to its per-claim verdicts and the
task model's full transcript.

Pass the harness. Bare chat completion is not what `submission/eval.py` does,
and a margin measured without the harness does not transfer to the one that
is scored.

### What the dashboard shows

| you want to know | where |
|---|---|
| is there any learning signal | reward curve, Summary tab |
| is GRPO getting gradient | score distribution + advantage spread per group |
| what the task model actually did | Rollouts tab, full text, or the conversation if you set `trajectory_messages` |
| is it reward hacking | sort rollouts by reward, read the top ones |
| where the wall clock went | step / substep timings |
| base vs trained | Evals tab, side by side |
| what was serving | Deployments tab |

**Custom metrics are free.** Any *numeric* key your reward or rollout function
puts on `sample.metadata` is charted per rollout automatically:

```python
sample.metadata["searches"] = n_searches      # becomes a curve
sample.metadata["claim_coverage"] = coverage  # becomes a curve
sample.metadata["trajectory_messages"] = msgs # becomes a conversation view
```

This is the cheapest reward-hacking detector you have. Chart the *behavior*
you believe drives the reward (searches issued, claims covered, answer length)
next to the reward itself: when reward climbs while the behavior collapses,
the model found a shortcut, and you can see it several steps before the dev
score tells you.

### Rules that still apply

1. H200 only (`gpu_type="H200"`), gpu_tools/README.md rule 2.
2. Append the `runs/GPU_LOG.jsonl` accounting line for every gym training
   run yourself (gpu_tools/README.md rule 5, the gym does not write your
   run's ledger). Same for long-lived gym eval deployments.
3. The judge is still the pinned one: `gym_eval` judges through
   `api_clients/judge_client`, the same `LEARNING_AGENT_JUDGE_URL` service as
   `rubric_eval`.

### What it does not show

The gym observes the *training run*. It has no GPU/CPU telemetry
(`system_monitor` in the observatory's record), no trace of the agent driving
the loop, no workspace snapshot, and no contamination audit. Those stay in
the run observatory, which ingests your run directory
exactly as before. Record the `training_run_id` in your `runs/LEARNING_LOG.jsonl` entry's
artifacts and the two views join up.

### Tutorial

The gym ships a worked example of this exact loop, corpus, self-authored rubric exam,
search harness, GRPO, margin, as a runnable notebook:
`tutorials/agent/` inside the pinned clone.
<!-- endif:rl,opd -->
