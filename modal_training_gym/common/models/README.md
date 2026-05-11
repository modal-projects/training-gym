# Models

Built-in `ModelConfig` presets shipped with Training Gym (Qwen3 family, etc.).
Each preset declares its HuggingFace repo, tokenizer config, and framework-
specific training presets (e.g. `SlimePreset`) with tuned parallelism and GPU
defaults that framework configs apply during `__post_init__`.

See [`base.py`](./base.py) for the base `ModelConfig` / `HFModelConfiguration`
classes and the individual `qwen3_*.py` files for each preset.
