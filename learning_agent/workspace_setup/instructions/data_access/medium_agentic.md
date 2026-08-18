## Your data and measurement

- `task/task_meta.yaml` states the task, its archetype, and the **scored
  surface** — which layers of the system your edits can move. Anything
  outside the surface is pinned on the operator's side.
- `task/corpus/`, when the task has one, is the domain study material.
  An environment task without a corpus is by design: the environment is
  the study material — roll out in it and learn from the trajectories.
- **No dev episode list is seeded on this track.** There are no gold
  answers either way — reward comes from the environment's own verifier —
  but the dev split's scenario ids and descriptions stay with the
  operator. Build your own study probes from the corpus and the
  environment.

`task/task.md` states the environment, the pinned protocol, and the exact
contract. The provided measurement instrument:

    # run YOUR submission on the operator's dev split through the pinned
    # environment (ids resolve on the service side; same env, simulator,
    # and reward as final scoring, reduced trial count).
    python toolbox/eval_tool/dev_eval.py --model /out/models/<tag>/merged

It packages your `submission/` (and the toolbox harness code), sends it
with your checkpoint to the operator's rollout service, and returns
per-episode rewards plus transcripts. Measurement is identical on every
track — what this track withholds is the local episode list, not the
signal. If a harness change did not move the dev number, it is not wired
into `submission/`.

The dev split is your iteration signal; the held-out test split measures
generalization. Memorizing dev episodes will not survive it.
