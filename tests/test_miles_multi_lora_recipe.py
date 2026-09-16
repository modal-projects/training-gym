"""MilesRecipe multi-LoRA (Tinker gateway) flags: only emitted when
``multi_lora_n_adapters`` is set, and the recipe rejects topologies the
gateway cannot serve before anything reaches Modal.
"""

import pytest

from modal_training_gym.common.models import Qwen3_30B
from modal_training_gym.train_recipes.miles_recipe import (
    MilesRecipe,
    Qwen3_30B_A3B_Tinker_Recipe,
)


def _gateway(**overrides) -> MilesRecipe:
    params = dict(
        multi_lora_n_adapters=2,
        lora_rank=8,
        colocate=False,
        actor_num_gpus_per_node=1,
        rollout_num_gpus=1,
    )
    params.update(overrides)
    return MilesRecipe(**params)


def _flags(recipe: MilesRecipe) -> list[str]:
    return recipe.cli_args()


def test_ordinary_recipe_emits_no_tinker_flags() -> None:
    flags = MilesRecipe().cli_args()
    assert "--multi-lora-n-adapters" not in flags
    assert not any("tinker" in flag for flag in flags)
    assert not MilesRecipe().is_tinker_gateway
    # 0 is Miles' own "disabled" value and stays ordinary training.
    assert not MilesRecipe(multi_lora_n_adapters=0).is_tinker_gateway


def test_negative_adapter_count_is_rejected() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        _gateway(multi_lora_n_adapters=-1)


def test_gateway_emits_multi_lora_and_tinker_flags() -> None:
    recipe = _gateway(
        tinker_server_port=10614,
        tinker_checkpoint_root="/checkpoints/tinker",
        tinker_base_model="my-base",
    )
    assert recipe.is_tinker_gateway
    flags = _flags(recipe)
    pairs = dict(zip(flags, flags[1:]))
    assert pairs["--multi-lora-n-adapters"] == "2"
    assert pairs["--lora-rank"] == "8"
    assert pairs["--tinker-server-port"] == "10614"
    assert pairs["--tinker-checkpoint-root"] == "/checkpoints/tinker"
    assert pairs["--tinker-base-model"] == "my-base"
    assert "--colocate" not in flags
    assert "--target-modules" not in flags


def test_train_groups_use_boolean_optional_action_spelling() -> None:
    flags = _flags(_gateway())
    assert "--tinker-train-attn" in flags
    assert "--tinker-train-mlp" in flags
    assert "--tinker-train-unembed" in flags
    assert not any(flag.startswith("--no-tinker-train") for flag in flags)

    flags = _flags(_gateway(tinker_train_unembed=False, tinker_train_mlp=False))
    assert "--tinker-train-attn" in flags
    assert "--no-tinker-train-mlp" in flags
    assert "--no-tinker-train-unembed" in flags
    assert "--tinker-train-mlp" not in flags
    assert "--tinker-train-unembed" not in flags


def test_unset_tinker_options_are_omitted() -> None:
    flags = _flags(_gateway())
    assert "--tinker-server-port" not in flags
    assert "--tinker-server-host" not in flags
    assert "--tinker-checkpoint-root" not in flags
    assert "--tinker-base-model" not in flags


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"colocate": True, "rollout_num_gpus": None}, "colocate"),
        ({"lora_rank": None}, "lora_rank"),
        ({"lora_rank": -8}, "lora_rank must be positive"),
        ({"target_modules": "q_proj,k_proj"}, "target_modules"),
        (
            {
                "tinker_train_attn": False,
                "tinker_train_mlp": False,
                "tinker_train_unembed": False,
            },
            "at least one",
        ),
        ({"pipeline_model_parallel_size": 2, "actor_num_gpus_per_node": 2}, "pipeline"),
        ({"context_parallel_size": 2, "actor_num_gpus_per_node": 2}, "context"),
        ({"qkv_format": "bshd"}, "qkv_format"),
        ({"experts_shared_outer_loras": True}, "experts_shared_outer_loras"),
        ({"optimizer": "muon"}, "optimizer"),
        ({"calculate_per_token_loss": True}, "calculate_per_token_loss"),
        ({"megatron_to_hf_mode": ""}, "bridge"),
        ({"async_mode": True}, "async_mode"),
        ({"train_backend": "fsdp"}, "train_backend"),
    ],
)
def test_gateway_rejects_unsupported_settings(overrides, message) -> None:
    # pydantic wraps the TrainingGymConfigError (a ValueError) in a ValidationError.
    with pytest.raises(ValueError, match=message):
        _gateway(**overrides)


def test_constraints_only_apply_to_gateway_recipes() -> None:
    # The same settings are legal for ordinary Miles training.
    MilesRecipe(colocate=True, target_modules="q_proj", qkv_format="bshd")


@pytest.mark.parametrize(
    "extra_config",
    [{"multi_lora_n_adapters": 2}, {"tinker_server_port": 1}],
)
def test_gateway_settings_are_rejected_in_extra_config(extra_config) -> None:
    # extra_config keys override same-named flags in Miles, which would let
    # them switch on the gateway without any of the checks above.
    with pytest.raises(ValueError, match="recipe fields"):
        MilesRecipe(extra_config=extra_config, lora_rank=8)


def test_gateway_recipe_is_rejected_by_train_launcher() -> None:
    from modal_training_gym.common.dataset import HuggingFaceDataset
    from modal_training_gym.frameworks.miles.launcher import build_miles_app

    with pytest.raises(ValueError, match="TrainConfig.train"):
        build_miles_app(
            training_run_id="run",
            miles=_gateway(),
            model=Qwen3_30B(),
            dataset=HuggingFaceDataset(
                hf_repo="org/data", input_column="prompt", output_column="answer"
            ),
        )


def test_qwen3_30b_a3b_preset_matches_upstream_example() -> None:
    recipe = Qwen3_30B_A3B_Tinker_Recipe()
    assert recipe.is_tinker_gateway
    assert recipe.model_config_class is Qwen3_30B
    flags = _flags(recipe)
    pairs = dict(zip(flags, flags[1:]))
    assert pairs["--multi-lora-n-adapters"] == "4"
    assert pairs["--lora-rank"] == "32"
    assert pairs["--lora-alpha"] == "64"
    assert pairs["--tensor-model-parallel-size"] == "2"
    assert pairs["--expert-model-parallel-size"] == "4"
    assert pairs["--actor-num-gpus-per-node"] == "4"
    assert pairs["--rollout-num-gpus"] == "4"
    assert pairs["--rollout-num-gpus-per-engine"] == "2"
    assert pairs["--sglang-lora-backend"] == "triton"
    assert pairs["--megatron-to-hf-mode"] == "bridge"
    assert recipe.miles_git_ref is not None
    assert "--no-gradient-accumulation-fusion" in flags
    assert "--colocate" not in flags
    assert recipe.total_nodes == 1


def test_preset_hf_checkpoint_comes_from_model() -> None:
    recipe = Qwen3_30B_A3B_Tinker_Recipe()
    assert "--hf-checkpoint" not in _flags(recipe)
    flags = recipe.cli_args(model=Qwen3_30B())
    pairs = dict(zip(flags, flags[1:]))
    assert pairs["--hf-checkpoint"] == "Qwen/Qwen3-30B-A3B"
    # miles_model_name sources the architecture, so no Megatron arch flags leak.
    assert "--num-layers" not in flags
