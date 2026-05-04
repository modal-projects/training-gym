## TODO
Tasks that are currently not fully validated.
- Add defaults to ensure each framework can be set up with <10 lines of code.
- Examples missing:
    - slime with LoRA
    - non-cluster examples with a smaller model to ensure cost savings.
    - using it on a larger model (e.g. GLM 4.7)
- In logs, tell user we are converting checkpoint from mt to hf. Do this in serve.
- Self host gym-server
- image_run_commands --> image_run_commands=lambda image: image.run_commands(
        "uv pip install --system aiohttp nltk>=3.8.0",
        "python -c \"import nltk; nltk.download('cmudict', quiet=True)\"",
    ),