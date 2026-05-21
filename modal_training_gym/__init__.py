from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "DatasetConfig": ("modal_training_gym.common.dataset", "DatasetConfig"),
    "HarborDataset": ("modal_training_gym.common.dataset", "HarborDataset"),
    "DeploymentConfig": ("modal_training_gym.common.deployment", "DeploymentConfig"),
    "EvalConfig": ("modal_training_gym.common.eval", "EvalConfig"),
    "EvalConfigDurable": ("modal_training_gym.common.eval", "EvalConfigDurable"),
    "EvalResult": ("modal_training_gym.common.eval", "EvalResult"),
    "EvalRowResult": ("modal_training_gym.common.eval", "EvalRowResult"),
    "GLM_4_7": ("modal_training_gym.common.models", "GLM_4_7"),
    "GLM_4_7_Flash": ("modal_training_gym.common.models", "GLM_4_7_Flash"),
    "HFModelConfiguration": (
        "modal_training_gym.common.models",
        "HFModelConfiguration",
    ),
    "HuggingFaceDataset": ("modal_training_gym.common.dataset", "HuggingFaceDataset"),
    "list_checkpoints": ("modal_training_gym.common.checkpoint", "list_checkpoints"),
    "METADATA_VOLUME_NAME": (
        "modal_training_gym.utils.metadata",
        "METADATA_VOLUME_NAME",
    ),
    "MetadataStore": ("modal_training_gym.utils.metadata", "MetadataStore"),
    "ModelArchitecture": ("modal_training_gym.common.models", "ModelArchitecture"),
    "ModelConfig": ("modal_training_gym.common.models", "ModelConfig"),
    "ModelDeployment": ("modal_training_gym.common.deployment", "ModelDeployment"),
    "MultiTurn": ("modal_training_gym.train_recipes.slime_recipe", "MultiTurn"),
    "Qwen3_0_6B": ("modal_training_gym.common.models", "Qwen3_0_6B"),
    "Qwen3_1_7B": ("modal_training_gym.common.models", "Qwen3_1_7B"),
    "Qwen3_4B": ("modal_training_gym.common.models", "Qwen3_4B"),
    "Qwen3_8B": ("modal_training_gym.common.models", "Qwen3_8B"),
    "Qwen3_14B": ("modal_training_gym.common.models", "Qwen3_14B"),
    "Qwen3_30B": ("modal_training_gym.common.models", "Qwen3_30B"),
    "Qwen3_32B": ("modal_training_gym.common.models", "Qwen3_32B"),
    "SlimeRecipe": ("modal_training_gym.train_recipes.slime_recipe", "SlimeRecipe"),
    "SlimeRecipeBlock": (
        "modal_training_gym.train_recipes.slime_recipe",
        "SlimeRecipeBlock",
    ),
    "TrainConfig": ("modal_training_gym.common.train", "TrainConfig"),
    "TrainResult": ("modal_training_gym.common.train_result", "TrainResult"),
    "setup": ("modal_training_gym.setup", "setup"),
    "WandbConfig": ("modal_training_gym.common.wandb", "WandbConfig"),
}

__all__ = [
    "DatasetConfig",
    "GLM_4_7",
    "GLM_4_7_Flash",
    "HarborDataset",
    "DeploymentConfig",
    "EvalConfig",
    "EvalConfigDurable",
    "EvalResult",
    "EvalRowResult",
    "HFModelConfiguration",
    "HuggingFaceDataset",
    "list_checkpoints",
    "METADATA_VOLUME_NAME",
    "MetadataStore",
    "ModelArchitecture",
    "ModelConfig",
    "ModelDeployment",
    "MultiTurn",
    "Qwen3_0_6B",
    "Qwen3_1_7B",
    "Qwen3_4B",
    "Qwen3_8B",
    "Qwen3_14B",
    "Qwen3_30B",
    "Qwen3_32B",
    "SlimeRecipe",
    "setup",
    "SlimeRecipeBlock",
    "TrainConfig",
    "TrainResult",
    "WandbConfig",
]


def __getattr__(name: str):
    module_name, attr_name = _EXPORTS.get(name, (None, None))
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
