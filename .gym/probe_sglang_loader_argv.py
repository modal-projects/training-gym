"""Check that the recipe's sglang weight-loader overrides survive miles' argv round-trip.

Runs miles' ``server_args_to_argv`` inside the recipe image with the
``extra_config`` sglang overrides applied the way ``miles_validate_args`` and
``get_server_args`` do, then parses the rendered argv back into ``ServerArgs``.
"""

import modal

if modal.is_local():
    from modal_training_gym.frameworks.miles.launcher import (
        _build_miles_base_image,
        apply_source_overlays,
    )
    from modal_training_gym.train_recipes.miles_recipe import (
        DeepSeek_V4_1_Flash_Recipe,
    )

    recipe = DeepSeek_V4_1_Flash_Recipe()
    image = apply_source_overlays(_build_miles_base_image(recipe), recipe)
    image = image.run_commands(*recipe.image_run_commands)
    overrides = dict(recipe.extra_config or {})
else:
    image = modal.Image.debian_slim()
    overrides = {}

app = modal.App("probe-dsv41-sglang-loader-argv", image=image)


@app.function(timeout=900)
def probe(overrides: dict) -> str:
    import json
    import os
    import subprocess

    script = r"""
import json, os, sys
sys.path.insert(0, "/root/miles")
from miles.backends.sglang_utils.server_args_utils import server_args_to_argv, parse_server_args_argv
overrides = json.loads(os.environ["OVERRIDES"])
kwargs = {"model_path": "/tmp/m", "trust_remote_code": True, "skip_tokenizer_init": True,
          "load_format": "dummy"}
for k, v in overrides.items():
    kwargs[k.removeprefix("sglang_")] = v
argv = server_args_to_argv(kwargs)
print("argv:", argv)
sa = parse_server_args_argv(argv)
print("disable_mmap:", sa.weight_loader_disable_mmap)
print("drop_cache:", sa.weight_loader_drop_cache_after_load)
print("extra_config:", repr(sa.model_loader_extra_config))
from sglang.srt.configs.load_config import LoadConfig
lc = LoadConfig(model_loader_extra_config=sa.model_loader_extra_config)
print("parsed extra_config:", lc.model_loader_extra_config)
"""
    os.makedirs("/tmp/m", exist_ok=True)
    json.dump(
        {
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "num_hidden_layers": 1,
            "vocab_size": 128,
            "max_position_embeddings": 128,
            "rms_norm_eps": 1e-6,
            "torch_dtype": "bfloat16",
        },
        open("/tmp/m/config.json", "w"),
    )
    env = dict(os.environ, OVERRIDES=json.dumps(overrides))
    r = subprocess.run(
        ["/opt/sglang/bin/python3", "-c", script],
        capture_output=True,
        text=True,
        env=env,
        cwd="/root/miles",
    )
    return r.stdout + "\n--- stderr (tail) ---\n" + r.stderr[-3000:]


@app.local_entrypoint()
def main():
    print(probe.remote(overrides))
