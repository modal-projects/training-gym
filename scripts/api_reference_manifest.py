"""
Curated manifest of public classes for API reference documentation.

Each entry maps a class to its module path, documentation group,
and class type (config_data or behavior).
"""

API_REFERENCE_MANIFEST = [
    # --- Core ---
    {
        "class_name": "ModelConfig",
        "module": "modal_training_gym.common.models.base",
        "group": "core",
        "class_type": "behavior",
        "sidebar_label": "ModelConfig",
    },
    {
        "class_name": "HFModelConfiguration",
        "module": "modal_training_gym.common.models.base",
        "group": "core",
        "class_type": "behavior",
        "sidebar_label": "HFModelConfiguration",
    },
    {
        "class_name": "ModelArchitecture",
        "module": "modal_training_gym.common.models.base",
        "group": "core",
        "class_type": "config_data",
        "sidebar_label": "ModelArchitecture",
    },
    {
        "class_name": "DatasetConfig",
        "module": "modal_training_gym.common.dataset",
        "group": "core",
        "class_type": "config_data",
        "sidebar_label": "DatasetConfig",
    },
    {
        "class_name": "HuggingFaceDataset",
        "module": "modal_training_gym.common.dataset",
        "group": "core",
        "class_type": "config_data",
        "sidebar_label": "HuggingFaceDataset",
    },
    {
        "class_name": "HarborDataset",
        "module": "modal_training_gym.common.dataset",
        "group": "core",
        "class_type": "config_data",
        "sidebar_label": "HarborDataset",
    },
    {
        "class_name": "WandbConfig",
        "module": "modal_training_gym.common.wandb",
        "group": "core",
        "class_type": "config_data",
        "sidebar_label": "WandbConfig",
    },
    {
        "class_name": "ModalRayCluster",
        "module": "modal_training_gym.common.ray_cluster",
        "group": "core",
        "class_type": "behavior",
        "sidebar_label": "ModalRayCluster",
    },
    {
        "class_name": "TrainResult",
        "module": "modal_training_gym.common.train_result",
        "group": "core",
        "class_type": "behavior",
        "sidebar_label": "TrainResult",
    },
    # --- Evaluation ---
    {
        "class_name": "EvalConfig",
        "module": "modal_training_gym.common.eval",
        "group": "evaluation",
        "class_type": "config_data",
        "sidebar_label": "EvalConfig",
    },
    {
        "class_name": "EvalResult",
        "module": "modal_training_gym.common.eval",
        "group": "evaluation",
        "class_type": "behavior",
        "sidebar_label": "EvalResult",
    },
    {
        "class_name": "EvalRowResult",
        "module": "modal_training_gym.common.eval",
        "group": "evaluation",
        "class_type": "behavior",
        "sidebar_label": "EvalRowResult",
    },
    # --- Models ---
    {
        "class_name": "Qwen3_0_6B",
        "module": "modal_training_gym.common.models.qwen3_0_6b",
        "group": "models",
        "class_type": "config_data",
        "sidebar_label": "Qwen3-0.6B",
    },
    {
        "class_name": "Qwen3_1_7B",
        "module": "modal_training_gym.common.models.qwen3_1_7b",
        "group": "models",
        "class_type": "config_data",
        "sidebar_label": "Qwen3-1.7B",
    },
    {
        "class_name": "Qwen3_4B",
        "module": "modal_training_gym.common.models.qwen3_4b",
        "group": "models",
        "class_type": "config_data",
        "sidebar_label": "Qwen3-4B",
    },
    {
        "class_name": "Qwen3_8B",
        "module": "modal_training_gym.common.models.qwen3_8b",
        "group": "models",
        "class_type": "config_data",
        "sidebar_label": "Qwen3-8B",
    },
    {
        "class_name": "Qwen3_14B",
        "module": "modal_training_gym.common.models.qwen3_14b",
        "group": "models",
        "class_type": "config_data",
        "sidebar_label": "Qwen3-14B",
    },
    {
        "class_name": "Qwen3_30B",
        "module": "modal_training_gym.common.models.qwen3_30b",
        "group": "models",
        "class_type": "config_data",
        "sidebar_label": "Qwen3-30B-A3B",
    },
    {
        "class_name": "Qwen3_32B",
        "module": "modal_training_gym.common.models.qwen3_32b",
        "group": "models",
        "class_type": "config_data",
        "sidebar_label": "Qwen3-32B",
    },
    # --- Training ---
    {
        "class_name": "TrainConfig",
        "module": "modal_training_gym.common.train",
        "group": "training",
        "class_type": "config_data",
        "sidebar_label": "TrainConfig",
    },
    {
        "class_name": "MultiTurn",
        "module": "modal_training_gym.train_recipes.slime_recipe.blocks",
        "group": "training",
        "class_type": "config_data",
        "sidebar_label": "MultiTurn",
    },
    {
        "class_name": "SlimeRecipe",
        "module": "modal_training_gym.train_recipes.slime_recipe.recipe",
        "group": "training",
        "class_type": "config_data",
        "sidebar_label": "SlimeRecipe",
    },
    # --- Deployment ---
    {
        "class_name": "DeploymentConfig",
        "module": "modal_training_gym.common.deployment",
        "group": "deployment",
        "class_type": "config_data",
        "sidebar_label": "DeploymentConfig",
    },
    {
        "class_name": "ModelDeployment",
        "module": "modal_training_gym.common.deployment",
        "group": "deployment",
        "class_type": "behavior",
        "sidebar_label": "ModelDeployment",
    },
    {
        "class_name": "SglangRecipe",
        "module": "modal_training_gym.deploy_recipes.sglang_recipe",
        "group": "deployment",
        "class_type": "config_data",
        "sidebar_label": "SglangRecipe",
    },
    {
        "class_name": "VllmRecipe",
        "module": "modal_training_gym.deploy_recipes.vllm_recipe",
        "group": "deployment",
        "class_type": "config_data",
        "sidebar_label": "VllmRecipe",
    },
]

GROUPS = {
    "core": {"label": "Core", "order": 1},
    "evaluation": {"label": "Evaluation", "order": 2},
    "models": {"label": "Models", "order": 3},
    "training": {"label": "Training", "order": 4},
    "deployment": {"label": "Deployment", "order": 5},
}


def class_to_reference_path(class_name: str) -> str | None:
    """Return the Starlight reference path for a class, or None if not in manifest."""
    for entry in API_REFERENCE_MANIFEST:
        if entry["class_name"] == class_name:
            slug = class_name.lower()
            return f"/reference/{entry['group']}/{slug}/"
    return None


CLASS_REFERENCE_PATHS = {
    entry["class_name"]: f"/reference/{entry['group']}/{entry['class_name'].lower()}/"
    for entry in API_REFERENCE_MANIFEST
}
