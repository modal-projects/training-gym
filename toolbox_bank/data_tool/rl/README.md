# RL data

GRPO needs two things from you: a prompts file and a reward function. Both
are yours to write.

## The prompts file

A JSON list; one row per question (see `rl_prompts.example.json`):

    {"question": "<what the task model is asked>",
     "label": {"question": "<same text>",
               "gold_answer": "<reference answer>",
               "rubric": [{"claim_id": "c1", "weight": 1,
                           "statement": "<what a correct answer must contain>"}]}}

The trainer turns `question` into the prompt. `label` never enters the
prompt — it goes to your reward function as `sample.label`. Build rows from
any QA data you have (gen_eval output, dev.json, synthetic pools).

## The reward

Write `async def rm(args, sample, **kwargs) -> float` and pass it to the
Training Gym recipe as `custom_rm_function` (see
`training_tool/training_gym/skills/` and `training_tool/README.md`).
How you score is your choice:

1. LLM-as-judge with the pinned gpt-5.6-luna. One call:

       from api_clients import judge_client
       score = judge_client.judge_claims(question, answer, rubric,
                                         gold_answer=gold, n_votes=1)["claim_score"]

   Canonical when `LEARNING_AGENT_JUDGE_URL` is set; pass `LEARNING_AGENT_JUDGE_URL`/
   `LEARNING_AGENT_JUDGE_TOKEN` into the training container (gpu_launcher `--env`) or
   the calls fail. Seconds per call — budget group size.
2. Be the judge yourself: deterministic scoring you design — token overlap
   with `gold_answer`, claim keyword checks, format gates. Free and instant;
   usually the right default at RL throughput.
3. Either way: never use the task model to grade itself (reward hacking by
   construction), and put shaping (search bonuses, length penalties) on top
   of correctness, never inside it.
