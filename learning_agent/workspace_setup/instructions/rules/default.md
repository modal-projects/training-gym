1. There will be no user interaction. Operate autonomously.
2. Your wall-clock budget is stated at launch; `bash timer.sh` shows what remains.
3. You will NEVER see the held-out evaluation questions or answers. They are not in
   this repository. Do not search for them elsewhere: any access to held-out
   eval data invalidates the run.
4. Training data must derive from the task corpus and generations made inside
   this run, by the task model or by YOU directly: you may write training
   examples yourself, or point the data tools' `--base-url`/`--model` at your
   own model. Teacher-written data is allowed and normal. No external
   datasets. The "Your data and measurement" section states any further data
   constraints for your run.
5. Only fine-tune `<TASK_MODEL>`: every model that answers at evaluation time must
   be `<TASK_MODEL>` or a fine-tune of it. Do not substitute or download another
   pretrained model as the answering engine.
6. `submission/eval.py` must make no external LLM API calls at answer time: no
   Anthropic/OpenAI/etc. anywhere in the answer path. Serve your own weights
   through your own serving wiring. API judges are for your dev-time
   measurement only.
7. The submission command above must work when your run ends: exact contract, no
   manual steps. If it does not run, there is nothing to score.

Remember: never ask for feedback. Execute the actions that make the most sense to you.
We will run your submission on the held-out questions once you are done.
