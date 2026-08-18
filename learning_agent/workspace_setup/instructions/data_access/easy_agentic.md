## Your data and measurement

Your task folder holds what this run ships:

- `task/task_meta.yaml` states the task, its archetype, and the **scored
  surface** — which layers of the system your edits can move. Anything
  outside the surface is pinned on the operator's side; editing it changes
  nothing about your score.
- `task/dev.json` holds the dev-split episode ids — there are NO gold
  answers or rubrics: **the reward comes from the environment itself**
  (its verifier, not any judge's opinion of your prose).
- `task/corpus/`, when present, is the domain study material (e.g. the
  knowledge base or policy the environment's scenarios draw on). An
  environment task without a corpus is by design: the environment is the
  study material — roll out in it and learn from the trajectories.

`task/task.md` states the environment, the pinned protocol, and the exact
contract. The provided measurement instrument:

    # run YOUR submission on the dev split through the operator's pinned
    # environment (same env, same simulator, same reward as final scoring;
    # reduced trial count). --model = the checkpoint to serve.
    python toolbox/eval_tool/dev_eval.py --model /out/models/<tag>/merged

It packages your `submission/` (and the toolbox harness code), sends it with
your checkpoint to the operator's rollout service, and returns per-episode
rewards plus transcripts. What gets scored at the end is that same pipeline
on the held-out split — a dev number is the official instrument at reduced
trials, so improving it and improving the submission are the same act. If a
harness change did not move the dev number, it is not wired into
`submission/`.

The dev split is your iteration signal; the held-out test split measures
generalization. Memorizing dev episodes will not survive it.
