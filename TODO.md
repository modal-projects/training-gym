Fixes
- Remove any string cli arg pass-in, such as `recipe_args`.
- Add defaults to ensure each framework can be set up with <10 lines of code.
- When you call build_app, it can also give you a TrainResult handler for you to write evals. What is  
  the best way to do this?   