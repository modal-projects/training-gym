## Your data and measurement

- `task/task_meta.yaml` states the task, its archetype, and the **scored
  surface** — which layers of the system your edits can move. Anything
  outside the surface is pinned on the operator's side.
- **No corpus is seeded on this track.** `task/brief.md` specifies exactly
  which primary sources constitute the domain material and how to acquire
  them (pinned versions matter — the scored environment uses the pinned
  snapshot, so a drifted copy misleads you). Acquire it, place it at
  `task/corpus/` per the brief, and study it. An environment task with no
  corpus at all is by design: install the pinned environment per the brief
  and learn from rollouts in its TRAINING split — the dev/test games are
  the measurement surface and stay out of your training data.
- **No dev episode list is seeded either.** There are no gold answers
  either way — reward comes from the environment's own verifier — but the
  dev split's scenario ids and descriptions stay with the operator.

`task/task.md` states the environment, the pinned protocol, and the exact
contract. The provided measurement instrument:

    # run YOUR submission on the operator's dev split through the pinned
    # environment (ids resolve on the service side; same env, simulator,
    # and reward as final scoring, reduced trial count).
    python toolbox/eval_tool/dev_eval.py --model /out/models/<tag>/merged

Measurement is identical on every track — what this track withholds is
the study material and the episode list, not the signal. If a harness
change did not move the dev number, it is not wired into `submission/`.

The dev split is your iteration signal; the held-out test split measures
generalization. Memorizing dev episodes will not survive it.
