## TODO
Tasks that are currently not fully validated.
- Add defaults to ensure each framework can be set up with <10 lines of code.
- Examples missing:
    - slime with LoRA
    - non-cluster examples with a smaller model to ensure cost savings.
    - using it on a larger model (e.g. GLM 4.7)
- In logs, tell user we are converting checkpoint from mt to hf. Do this in serve.
- Use mount_tools_dir


## Unknowns
- How do u add tool calling to your model: eval is single turn in how it works right now
- EvalConfig passes in EvalFn, which is very similar to how we pass in custom_rm_fn, but that requires allowing the two to translate between each other, which is very cursed.
- We might want to allow users to specify top level args (e.g. number of rollouts) without knowing what framework they are using?


## Standouts/Differentiator
People care about 2 things:
- How efficient training is
- How customizable is the training

## Ideas
- Training with sandbox (deep multiturns agents using sandbox)