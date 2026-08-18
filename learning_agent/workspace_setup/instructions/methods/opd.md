- **On-policy distillation (OPD)**: a teacher's next-token distribution
  is the training signal; no rubric is needed. Two teachers are
  supported, pick per experiment:
  - a separate teacher endpoint, through the Training Gym SDK like RL:
    the same prompts-file format (`data_tool/rl/README.md`), slime's
    `on_policy_distillation` rollout, wired by the SDK's `opd_reward`
    module (see `toolbox/training_tool/README.md`).
  - the task model as its OWN teacher (on-policy self-distillation):
    the teacher copy answers with privileged context in its window, a
    corpus file or feedback, and the student learns to match without
    it. Runs through `toolbox/training_tool/self_distillation/`; the
    data-transform variant is described in
    `data_tool/self_distill/self_distill.md`.
  Background papers: SDFT (arxiv.org/abs/2601.19897), OPSD
  (arxiv.org/abs/2601.18734), SDPO (arxiv.org/abs/2601.20802).
