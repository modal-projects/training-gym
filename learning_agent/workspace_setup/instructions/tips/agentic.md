- Training needs roughly thousands to a few tens of thousands of
  examples or prompts. A few hundred rows is not enough signal.
- The TASK ENVIRONMENT is where training data comes from: run the task
  model in it and learn from the trajectories, whatever the method.
  The dev set is for understanding the task and measuring progress,
  not the thing to build data around.
- Always dedup/decontaminate before training.
