"""
Curated manifest of public classes for API reference documentation.

Each entry maps a class to its module path, documentation group,
and class type (config_data or behavior).
"""

API_REFERENCE_MANIFEST = [
    # --- Core ---
    {
        "class_name": "ModelConfiguration",
        "module": "modal_training_gym.common.models.base",
        "group": "core",
        "class_type": "behavior",
        "sidebar_label": "ModelConfiguration",
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
        "class_name": "LlmJudge",
        "module": "modal_training_gym.common.llm_judge",
        "group": "core",
        "class_type": "behavior",
        "sidebar_label": "LlmJudge",
    },
    # --- Models ---
    {
        "class_name": "Qwen3_4B",
        "module": "modal_training_gym.common.models.qwen3_4b",
        "group": "models",
        "class_type": "config_data",
        "sidebar_label": "Qwen3-4B",
    },
    {
        "class_name": "Qwen3_32B",
        "module": "modal_training_gym.common.models.qwen3_32b",
        "group": "models",
        "class_type": "config_data",
        "sidebar_label": "Qwen3-32B",
    },
    {
        "class_name": "GLM_4_7",
        "module": "modal_training_gym.common.models.glm_4_7",
        "group": "models",
        "class_type": "config_data",
        "sidebar_label": "GLM-4.7",
    },
    {
        "class_name": "Llama2_7B",
        "module": "modal_training_gym.common.models.llama2_7b",
        "group": "models",
        "class_type": "config_data",
        "sidebar_label": "Llama2-7B",
    },
    {
        "class_name": "Kimi_K2_5",
        "module": "modal_training_gym.common.models.kimi_k2_5",
        "group": "models",
        "class_type": "config_data",
        "sidebar_label": "Kimi-K2.5",
    },
    # --- Frameworks: slime ---
    {
        "class_name": "SlimeConfig",
        "module": "modal_training_gym.frameworks.slime.config",
        "group": "frameworks",
        "class_type": "config_data",
        "sidebar_label": "SlimeConfig",
    },
    {
        "class_name": "ModalConfig",
        "module": "modal_training_gym.frameworks.slime.config",
        "group": "frameworks",
        "class_type": "config_data",
        "sidebar_label": "ModalConfig (slime)",
    },
    # --- Frameworks: MS-SWIFT ---
    {
        "class_name": "MsSwiftFrameworkConfig",
        "module": "modal_training_gym.frameworks.ms_swift.config",
        "group": "frameworks",
        "class_type": "config_data",
        "sidebar_label": "MsSwiftFrameworkConfig",
    },
    {
        "class_name": "MsSwiftConfig",
        "module": "modal_training_gym.frameworks.ms_swift.config",
        "group": "frameworks",
        "class_type": "config_data",
        "sidebar_label": "MsSwiftConfig",
    },
    # --- Frameworks: Miles ---
    {
        "class_name": "MilesFrameworkConfig",
        "module": "modal_training_gym.frameworks.miles.config",
        "group": "frameworks",
        "class_type": "config_data",
        "sidebar_label": "MilesFrameworkConfig",
    },
    {
        "class_name": "MilesConfig",
        "module": "modal_training_gym.frameworks.miles.config",
        "group": "frameworks",
        "class_type": "config_data",
        "sidebar_label": "MilesConfig",
    },
    # --- Frameworks: Harbor ---
    {
        "class_name": "HarborFrameworkConfig",
        "module": "modal_training_gym.frameworks.harbor.config",
        "group": "frameworks",
        "class_type": "config_data",
        "sidebar_label": "HarborFrameworkConfig",
    },
    {
        "class_name": "HarborConfig",
        "module": "modal_training_gym.frameworks.harbor.config",
        "group": "frameworks",
        "class_type": "config_data",
        "sidebar_label": "HarborConfig",
    },
    # --- Results ---
    {
        "class_name": "TrainResult",
        "module": "modal_training_gym.common.train_result",
        "group": "core",
        "class_type": "behavior",
        "sidebar_label": "TrainResult",
    },
]

GROUPS = {
    "core": {"label": "Core", "order": 1},
    "models": {"label": "Models", "order": 2},
    "frameworks": {"label": "Frameworks", "order": 3},
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
