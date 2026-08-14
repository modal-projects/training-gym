# The Learning Toolbox

Tools and training packages for making the task model expert at a task.
The research cycle itself is in AGENTS.md; this file is the reference:
the structure, the rules, and the generated catalog. Each category's
README documents its own shelf. Everything here is yours to rewrite:
change harnesses, invent tools, delete what you don't need.

## Structure

```
toolbox/
  TOOLS.md            this file (rules + generated catalog)
  repos.yaml          package pins        clone_repos.py   materializes them
  gpu_tools/          Modal GPUs: gpu_launcher.py (run any command on a GPU)
                      gpu_sandbox.py (interactive SSH box)  modal.md
                      training.md  sandbox.md  (GPU rules in its README)
  api_clients/        LLM + judge API clients (library, imported not run)

  data_tool/          makes training data; one folder per family, seeded
                      for YOUR training methods. METHOD CARDS (.md), not
                      scripts — you implement them
    pool/             dedup_decontam  mix (runnable tools — the hygiene gate)

  training_tool/      the training packages seeded for YOUR task (its
                      README maps each method to its package)

  eval_tool/          gen_eval  rubric_eval  gym_eval
  harness_tool/       the harness starters seeded for YOUR task
                      (each file's docstring says what it is)
  inference_tool/     vllm_serve  sglang_serve
```

The toolbox holds three kinds of things:

- **Training packages** (`training_tool/`): full cloned repos, pinned in
  `repos.yaml`. Drive them directly through their own docs and commands.
- **Method cards** (`.md` files in `data_tool/`): recipes for data-making
  methods — when to use one, the I/O contract, the known traps. Cards are
  not executable; making data is YOUR capability. Read the card, then
  implement it your way as a `.py` under `data_tool/`. The cards are
  starting points, not a menu: vary them, combine them, invent methods
  they don't cover. A new method needs no registration — it is judged the
  same way as everything else, dev margin over the untrained base.
- **Runnable tools** (the `.py` files everywhere else): one file each.
  The docstring says when to use it; `--help` documents the flags.

When you write a new tool, drop the file into the category it belongs to
and it is a tool: your run's timeline picks up every executed toolbox
path automatically. The catalog below lists what this workspace was
seeded with; tools you add are picked up without any registration.

Run `data_tool/pool/dedup_decontam` on every dataset before training
(duplicates and dev-set leakage poison the margin; it hard-fails without
`--eval-questions`), and blend finished datasets with
`data_tool/pool/mix`.

## Rules

1. A tool goes where the thing it changes lives: training data ->
   `data_tool`, weights -> `training_tool`, scores -> `eval_tool`, the
   answering program -> `harness_tool`, serving -> `inference_tool`.
2. The task agent answers through EXACTLY ONE harness; `submission/eval.py`
   is that choice made executable.
3. Contracts: data tools read and write training examples as JSONL, one
   example per line, in the shapes the method cards in `data_tool/`
   specify; trainers write
   `/out/models/$LEARNING_AGENT_RUN_ID/<tag>/merged`;
   evals write JSON with `mean` and a confidence interval.
4. Judging: with `LEARNING_AGENT_JUDGE_URL` set (workspaces have it), `eval_tool/*` and
   the judge CLI run on the SAME pinned judge that scores your submission.
   Use it freely wherever you need an LLM judge — eval, answer ranking,
   data filtering, training rewards.
5. The judge is always the big pinned model, never the task model judging
   itself. Answer-time API calls stay forbidden.
6. GPU work: compose the command, run it with `toolbox/gpu_tools/gpu_launcher.py`
   (`--pip-e <package dir>` installs a library package into the job image).

<!-- CATALOG BELOW IS GENERATED at seeding from the tools in THIS workspace — do not edit -->
