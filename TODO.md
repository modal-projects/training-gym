Fixes
- Remove any string cli arg pass-in, such as `recipe_args`.
- Add defaults to ensure each framework can be set up with <10 lines of code.
- When you call build_app, it can also give you a TrainResult handler for you to write evals. What is  
  the best way to do this?   
- Examples missing:
    - ms-swift with LoRA
    - slime/miles with LoRA
    - Using it on a larger model (e.g. GLM 4.7 or Kimi K2.5), what does it mean to train a larger model? Be more general :D