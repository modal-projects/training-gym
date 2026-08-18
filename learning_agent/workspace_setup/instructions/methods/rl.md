- **RL (GRPO)**: through the Training Gym SDK
  (`toolbox/training_tool/training_gym/`;
  github.com/modal-projects/training-gym); its `skills/` folder holds
  step-by-step playbooks, so read the relevant skill before driving it.
  The RL workflow is yours end to end: generate questions from the
  corpus, write grading rubrics for them, set up the RL task (a prompts
  file plus a reward function that judges each rollout against your
  rubric, with the pinned gpt-5.6-luna API as LLM-as-judge via
  `toolbox/api_clients/judge_client.py`, the same judge that scores
  your submission), launch the GRPO run, and monitor it on the gym's
  live dashboard (rewards, advantages, rollouts) while it trains.
  Plan for at least one GRPO experiment during your run rather than
  spending the whole budget on other training and harness tweaks.
