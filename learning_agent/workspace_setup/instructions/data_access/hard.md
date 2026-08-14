## Your data and measurement

No corpus is provided for this run. `task/brief.md` specifies exactly
which primary sources constitute the task corpus. Acquire them and
place the normalized corpus at `task/corpus/` (layout per
`task/task.md`); every provided tool reads that path. Acquire exactly
what the brief pins: versions and document lists matter.

Primary sources only: the project's own repository, documentation, or
filings named in the brief. No third-party question/answer, forum, or
discussion data as training supervision.

No gold answers are provided either; building your own measurement is
part of the task. Standard route: author your own dev questions with
gold answers drawn from the corpus (see `toolbox/eval_tool/gen_eval.py`).
The judge instrument accepts any gold file:

    python toolbox/eval_tool/rubric_eval.py --dev <your_gold.json> \
        --answers <answers.json> --task <TASK> --out <results.json>

Each row needs `id`, `question`, and `rubric` (a weighted list of
claims); `gold_answer` is optional: the judge scores against it as the
reference answer when present, and without one otherwise.

Compare candidates on the same questions, and hold your own eval
questions out of training data.
