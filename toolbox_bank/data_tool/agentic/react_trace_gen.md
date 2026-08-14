# react_trace_gen — real-observation ReAct traces for trace-mode SFT

Method card: you implement this (write your own generator under `data_tool/`),
the card tells you the contract, the recipe, and the traps.

## When to use

Your track's harness searches the corpus (grep/glob/read) before answering,
and you want the task model to LEARN that search behavior. The internalization
generators teach closed-book recall; this teaches the tool loop itself.

## Contract

Output JSONL, one multi-turn trace per row — the shape trainers loss-mask to
assistant turns:

    {"messages": [
        {"role": "system",    "content": "<the EXACT system prompt your harness uses>"},
        {"role": "user",      "content": "<question>"},
        {"role": "assistant", "content": "ACTION: {\"tool\": \"glob\", ...}"},
        {"role": "user",      "content": "OBSERVATION:\n<real corpus output>"},
        ...,
        {"role": "assistant", "content": "FINAL: <gold answer>"}]}

## Recipe

1. Take QA rows WITH evidence (grounded_qa output: `{question, answer,
   evidence:[{path, start_line, end_line}]}`).
2. Plan a discovery sequence per row — glob(dir) -> grep(salient token from
   the evidence) -> read_file(the exact evidence lines) -> FINAL(gold).
3. EXECUTE every action against the corpus for real, and put the REAL output
   in each OBSERVATION turn. Fabricated observations teach the task model to
   expect outputs the tools never produce.
4. (Optional, riskier) model mode: let the served model propose the actions,
   execute them for real, fall back to the evidence plan when observations
   come back empty.

## Pitfalls

- Grammar drift is the whole game: the system prompt, the `ACTION:` JSON
  schema, and the `OBSERVATION:` framing must match your answering harness
  EXACTLY (the frozen eval grammar in `harness/eval.py`). Diff your trace
  turns against a live harness rollout before training on thousands of them.
- Sandbox the corpus tools you execute: absolute paths and `..` escape a
  naive `os.path.join(root, pattern)`, a `startswith` prefix check without a
  trailing separator lets `corpus/../corpus_x` through, and symlinks need
  `realpath`. In model mode you are executing model-proposed actions — treat
  them as untrusted.
- Reject traces with empty observations (stale evidence line ranges after a
  corpus change produce `OBSERVATION:` followed by nothing) — keeping them
  teaches the task model to assert FINAL from no evidence. Watch for the subtle
  empty: a read past EOF that returns "" is just as poisonous as a "[no
  matches]" marker.
- Cap model-proposed regexes (timeout / length) — catastrophic backtracking
  on a pathological pattern hangs the generator.
- Then `pool/dedup_decontam.py`, like every pool.

Starting point, not a menu: vary it, combine it, or replace it with a better
method — new methods are scored like everything else, dev margin over base.
