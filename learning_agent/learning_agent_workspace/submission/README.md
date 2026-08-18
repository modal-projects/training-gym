# submission

The scoring surface of the benchmark. The agent's deliverable is its whole
working copy of this repo; this directory is the part post-eval drives. Three
files, two of them the real contract:

| file | role | contract that must survive your edits |
|---|---|---|
| `serve.py` | the trained task model behind an OpenAI-compatible endpoint | `ensure_endpoint(weights, base_url, port) -> (base_url, model)`; set `WEIGHTS` |
| `agent.py` | **build() -> the policy** post-eval calls | `build(**overrides) -> Agent` with `answer(q) -> {"answer": str}` · `answer_batch([{id,question}]) -> {id: answer}` · `act(instruction, execute, driver=) -> result dict` · `tool_turn(messages, tools, execute_tool) -> (text, messages)` |
| `eval.py` | thin QA CLI over `agent.build()` (kept for back-compat) | `python submission/eval.py --input questions.json --output answers.json` — input: JSON array of `{"id", "question"}`; output: `{id: answer}` for every id |

Rewrite your harness in `agent.py` (and your serving in `serve.py`) — post-eval
imports `build()` directly for agentic tasks and goes through the `eval.py` CLI
for QA, so either way it scores exactly the object you developed against.

## How each archetype is scored (operator)

**QA (dspy / openclaw / fav2 / maud)** — questions held out; answers judged
against hidden gold + rubrics:

```bash
python submission/eval.py --input <heldout-questions.json> --output answers.json
python toolbox/eval_tool/rubric_eval.py --dev <heldout-gold.json> \
    --answers answers.json --task <task> --out results.json
```

**Agentic (alfworld / tau2_*)** — no questions file; the operator's
`harness/rollout.py` builds the agent and drives it against the scored
environment through the task's adapter, injecting the only bridge
(`execute` / `execute_tool`) itself:

```python
from submission.agent import build
agent = build()
agent.act(instruction, execute=env_step, driver="react")  # env episode; env's
                                                          # verifier sets reward
agent.tool_turn(messages, tools, execute_tool=env_tool)   # tau2 dev-time turn
                                                          # (official: native runner)
```

The driver (`react` default, `mini_swe` for terminal tasks) is pinned per task
by the operator's `task_configs/<task>.yaml` under `agent:` (the protocol is
restated in `task/task.md`). The per-archetype minimal agents
(`react_env_agent`, `mini_swe_agent`, `react_tool_agent`) ship into
`toolbox/harness_tool/` — a QA workspace carries the QA starters
(`react_loop`, `completion_qa`) there instead
(see `workspace_setup/prepare_workspace.sh`).

## Notes

- Baseline `agent.py` answers closed-book and runs the stock driver loops.
  Checkpoints on the Modal volume must be served by the agent's own wiring
  (that is part of the harness it submits).
- `--backend mock` / `build(backend="mock")` exercises the contract offline
  (deterministic stub; used by the test suite — not a real run).
- No external LLM APIs at answer/act time (AGENTS.md rule 6): every
  acting model must be the designated task model base or a fine-tune of it, served
  by the submission's own wiring. The in-repo `harness/eval.py` (fixed ReAct
  search + pinned judge) remains available as a dev-signal reference instrument.
