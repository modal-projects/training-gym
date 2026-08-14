- GPU jobs run on Modal (already authenticated); up to 8x H200-class
  GPUs. Long GPU jobs: launch detached and poll; do not block silently
  on one wait.
- The untrained task model's weights are in this workspace at `model/`
  (`<TASK_MODEL>`, pinned revision).
- Dev-time evaluation is yours to run, but use the pinned judge to run
  it: when `LEARNING_AGENT_JUDGE_URL` is set in `.env`, the provided judge
  instruments (`toolbox/eval_tool/`) reach the operator's judge service,
  the SAME pinned judge model that will score your submission
  (`canonical:true`). Use it for dev-set evals and for every
  intermediate checkpoint you consider, so your numbers predict the
  official ones. You are free to use it any time you need an LLM judge:
  ranking answers, filtering generated data, training rewards. It
  grades; it is budgeted per session and is not a general
  text-generation API. Without it, the instruments fall back to the
  local `claude` CLI (stamped `canonical:false`, a weaker signal).
  Whatever you measure with, compare like with like.
