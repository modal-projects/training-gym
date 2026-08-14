## Your data and measurement

`task/corpus/` is the training corpus, the raw domain material your
training data comes from. No gold answers are provided for this run.
`task/task.md` describes the corpus and the expected answer format;
building your own measurement is part of the task.

Standard route: author your own dev questions with gold answers drawn
from the corpus (see `toolbox/eval_tool/gen_eval.py`). The judge
instrument accepts any gold file:

    python toolbox/eval_tool/rubric_eval.py --dev <your_gold.json> \
        --answers <answers.json> --task <TASK> --out <results.json>

Each row needs `id`, `question`, and `rubric` (a weighted list of
claims); `gold_answer` is optional: the judge scores against it as the
reference answer when present, and without one otherwise. `topic` and
`evidence` may be included for your own bookkeeping but are not read by
this instrument.

Compare candidates on the same questions, and hold your own eval
questions out of training data.
