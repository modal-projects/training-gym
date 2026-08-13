from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "Checkpoint": ("modal_training_gym.common.checkpoint", "Checkpoint"),
    "CheckpointType": ("modal_training_gym.common.checkpoint", "CheckpointType"),
    "convert_checkpoint_to_hf": (
        "modal_training_gym.common.checkpoint",
        "convert_checkpoint_to_hf",
    ),
    "CustomDeployment": (
        "modal_training_gym.common.deployment",
        "CustomDeployment",
    ),
    "DatasetConfig": ("modal_training_gym.common.dataset", "DatasetConfig"),
    "Endpoint": ("modal_training_gym.common.endpoint", "Endpoint"),
    "HarborDataset": ("modal_training_gym.common.dataset", "HarborDataset"),
    "EvalConfig": ("modal_training_gym.common.eval", "EvalConfig"),
    "EvalConfigDurable": ("modal_training_gym.common.eval", "EvalConfigDurable"),
    "EvalResult": ("modal_training_gym.common.eval", "EvalResult"),
    "EvalRowResult": ("modal_training_gym.common.eval", "EvalRowResult"),
    "GpuAllocationError": ("modal_training_gym.common.errors", "GpuAllocationError"),
    "AudioEvalRowResult": ("modal_training_gym.common.eval", "AudioEvalRowResult"),
    "ImageEvalRowResult": ("modal_training_gym.common.eval", "ImageEvalRowResult"),
    "Sample": ("modal_training_gym.common.sample", "Sample"),
    "extract_code": ("modal_training_gym.common.eval", "extract_code"),
    "HarborEval": ("modal_training_gym.common.eval", "HarborEval"),
    "GLM_4_7": ("modal_training_gym.common.models", "GLM_4_7"),
    "HFModelConfiguration": (
        "modal_training_gym.common.models",
        "HFModelConfiguration",
    ),
    "HuggingFaceDataset": ("modal_training_gym.common.dataset", "HuggingFaceDataset"),
    "MultimodalDataset": ("modal_training_gym.common.dataset", "MultimodalDataset"),
    "list_checkpoints": ("modal_training_gym.common.checkpoint", "list_checkpoints"),
    "METADATA_VOLUME_NAME": (
        "modal_training_gym.utils.metadata",
        "METADATA_VOLUME_NAME",
    ),
    "MetadataStore": ("modal_training_gym.utils.metadata", "MetadataStore"),
    "ModalCaptureError": (
        "modal_training_gym.common.modal_refs",
        "ModalCaptureError",
    ),
    "ModelArchitecture": ("modal_training_gym.common.models", "ModelArchitecture"),
    "ModelConfig": ("modal_training_gym.common.models", "ModelConfig"),
    "Kimi_K2_5": ("modal_training_gym.common.models", "Kimi_K2_5"),
    "Kimi_K2_6": ("modal_training_gym.common.models", "Kimi_K2_6"),
    "Kimi_K2_5_LoRA_Recipe": (
        "modal_training_gym.train_recipes.miles_recipe",
        "Kimi_K2_5_LoRA_Recipe",
    ),
    "Kimi_K2_6_LoRA_Recipe": (
        "modal_training_gym.train_recipes.miles_recipe",
        "Kimi_K2_6_LoRA_Recipe",
    ),
    "MilesRecipe": ("modal_training_gym.train_recipes.miles_recipe", "MilesRecipe"),
    "parse_qwen3_response": (
        "modal_training_gym.common.models",
        "parse_qwen3_response",
    ),
    "ParsedResponse": ("modal_training_gym.common.models", "ParsedResponse"),
    "Qwen3_0_6B": ("modal_training_gym.common.models", "Qwen3_0_6B"),
    "Qwen3_1_7B": ("modal_training_gym.common.models", "Qwen3_1_7B"),
    "Qwen3_4B": ("modal_training_gym.common.models", "Qwen3_4B"),
    "Qwen3_4b_Recipe": (
        "modal_training_gym.train_recipes.slime_recipe",
        "Qwen3_4b_Recipe",
    ),
    "Qwen3_5_0_8B": ("modal_training_gym.common.models", "Qwen3_5_0_8B"),
    "Qwen3_5_0_8b_Recipe": (
        "modal_training_gym.train_recipes.slime_recipe",
        "Qwen3_5_0_8b_Recipe",
    ),
    "Qwen3_5_2B": ("modal_training_gym.common.models", "Qwen3_5_2B"),
    "Qwen3_5_2b_Recipe": (
        "modal_training_gym.train_recipes.slime_recipe",
        "Qwen3_5_2b_Recipe",
    ),
    "Qwen3_5_4B": ("modal_training_gym.common.models", "Qwen3_5_4B"),
    "Qwen3_5_4b_Recipe": (
        "modal_training_gym.train_recipes.slime_recipe",
        "Qwen3_5_4b_Recipe",
    ),
    "Qwen3_5_9B": ("modal_training_gym.common.models", "Qwen3_5_9B"),
    "Qwen3_5_9b_Recipe": (
        "modal_training_gym.train_recipes.slime_recipe",
        "Qwen3_5_9b_Recipe",
    ),
    "Qwen3_8B": ("modal_training_gym.common.models", "Qwen3_8B"),
    "Qwen3_30B": ("modal_training_gym.common.models", "Qwen3_30B"),
    "Qwen3_6_35B": ("modal_training_gym.common.models", "Qwen3_6_35B"),
    "Qwen3_6_27B": ("modal_training_gym.common.models", "Qwen3_6_27B"),
    "Qwen3_6_27b_Recipe": (
        "modal_training_gym.train_recipes.slime_recipe",
        "Qwen3_6_27b_Recipe",
    ),
    "Qwen3_VL_8B": ("modal_training_gym.common.models", "Qwen3_VL_8B"),
    "Qwen3_VL_8b_Recipe": (
        "modal_training_gym.train_recipes.slime_recipe",
        "Qwen3_VL_8b_Recipe",
    ),
    "Qwen3_ASR_1_7B": ("modal_training_gym.common.models", "Qwen3_ASR_1_7B"),
    "Qwen3_ASR_1_7b_Recipe": (
        "modal_training_gym.train_recipes.slime_recipe",
        "Qwen3_ASR_1_7b_Recipe",
    ),
    "score_in_sandbox": ("modal_training_gym.common.eval", "score_in_sandbox"),
    "SlimeRecipe": ("modal_training_gym.train_recipes.slime_recipe", "SlimeRecipe"),
    "StitchRecipe": ("modal_training_gym.train_recipes.stitch_recipe", "StitchRecipe"),
    "StitchServeConfig": (
        "modal_training_gym.train_recipes.stitch_recipe",
        "StitchServeConfig",
    ),
    "StitchTrainConfig": (
        "modal_training_gym.train_recipes.stitch_recipe",
        "StitchTrainConfig",
    ),
    "ToolCall": ("modal_training_gym.common.models", "ToolCall"),
    "TrainConfig": ("modal_training_gym.common.train", "TrainConfig"),
    "TrainingGymConfigError": (
        "modal_training_gym.common.errors",
        "TrainingGymConfigError",
    ),
    "TrainingGymError": ("modal_training_gym.common.errors", "TrainingGymError"),
    "TrainingGroup": ("modal_training_gym.common.training_group", "TrainingGroup"),
    "TrainingRun": ("modal_training_gym.common.run", "TrainingRun"),
    "TrainResult": ("modal_training_gym.common.train_result", "TrainResult"),
    "WandbConfig": ("modal_training_gym.common.wandb", "WandbConfig"),
}

__all__ = [
    "CustomDeployment",
    "Kimi_K2_6_LoRA_Recipe",
    "Kimi_K2_5_LoRA_Recipe",
    "Checkpoint",
    "CheckpointType",
    "convert_checkpoint_to_hf",
    "DatasetConfig",
    "Endpoint",
    "GLM_4_7",
    "HarborDataset",
    "EvalConfig",
    "EvalConfigDurable",
    "EvalResult",
    "EvalRowResult",
    "GpuAllocationError",
    "AudioEvalRowResult",
    "ImageEvalRowResult",
    "Sample",
    "extract_code",
    "HarborEval",
    "HFModelConfiguration",
    "HuggingFaceDataset",
    "MultimodalDataset",
    "list_checkpoints",
    "Kimi_K2_6",
    "Kimi_K2_5",
    "METADATA_VOLUME_NAME",
    "MetadataStore",
    "ModalCaptureError",
    "ModelArchitecture",
    "ModelConfig",
    "MilesRecipe",
    "parse_qwen3_response",
    "ParsedResponse",
    "Qwen3_0_6B",
    "Qwen3_1_7B",
    "Qwen3_4B",
    "Qwen3_4b_Recipe",
    "Qwen3_5_0_8B",
    "Qwen3_5_0_8b_Recipe",
    "Qwen3_5_2B",
    "Qwen3_5_2b_Recipe",
    "Qwen3_5_4B",
    "Qwen3_5_4b_Recipe",
    "Qwen3_5_9B",
    "Qwen3_5_9b_Recipe",
    "Qwen3_8B",
    "Qwen3_30B",
    "Qwen3_6_35B",
    "Qwen3_6_27B",
    "Qwen3_6_27b_Recipe",
    "Qwen3_ASR_1_7b_Recipe",
    "Qwen3_VL_8b_Recipe",
    "score_in_sandbox",
    "SlimeRecipe",
    "StitchRecipe",
    "StitchServeConfig",
    "StitchTrainConfig",
    "ToolCall",
    "TrainConfig",
    "TrainingGymConfigError",
    "TrainingGymError",
    "TrainingGroup",
    "TrainingRun",
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
