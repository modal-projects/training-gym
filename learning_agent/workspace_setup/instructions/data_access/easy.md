## Your data and measurement

Your task folder holds both data sources for this run:

- `task/corpus/` is the training corpus, the raw domain material your
  training data comes from.
- `task/dev.json` holds dev questions WITH gold answers and grading
  rubrics, your only ground truth. The dev set shows you what the task
  looks like: the style of question, the expected answer format, and the
  kinds of knowledge and reasoning the evaluation draws on. Tune with
  it, but do not over-fit to it: changes that only chase the dev score,
  such as rules hardcoded for those particular questions or outputs
  shaped to those particular gold answers, will not survive the held-out
  set.

`task/task.md` describes the corpus and answer format. The provided
measurement instrument (use it, or build better ones):

    # judge any answers JSON against dev gold; measures submission/eval.py end to end
    python toolbox/eval_tool/rubric_eval.py --dev task/dev.json \
        --answers <answers.json> --task <TASK> --out <results.json>

Measure through your FULL harness: produce the answers with
`submission/eval.py` (or the same policy it wires), then judge them.
What gets scored at the end is that pipeline, not the bare model. With
`LEARNING_AGENT_JUDGE_URL` set (your `.env` has it) the verdicts come from the SAME
pinned judge that scores your submission, so your dev numbers and the
official score are one instrument.

The dev set may guide your decisions but must never appear in training
data.
