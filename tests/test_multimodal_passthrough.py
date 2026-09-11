import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from modal_training_gym.common.dataset import HuggingFaceDataset, MultimodalDataset
from modal_training_gym.common.errors import TrainingGymConfigError
from modal_training_gym.common.launcher_utils import prepare_launch_config
from modal_training_gym.common.models import (
    Gemma4_26B_A4B,
    Inkling_Small,
    Qwen3_4B,
    Qwen3_5_4B,
    Qwen3_6_27B,
    Qwen3_6_35B,
    Qwen3_8_27B,
    Qwen3_ASR_1_7B,
    Qwen3_VL_8B,
)
from modal_training_gym.common.train import TrainConfig
from modal_training_gym.frameworks.slime.launcher import _apply_qwen35_vl_image_train
from modal_training_gym.train_recipes.miles_recipe.gemma4_26b_a4b import (
    Gemma4_26B_A4B_Recipe,
)
from modal_training_gym.train_recipes.miles_recipe.inkling import (
    Inkling_Small_LoRA_Recipe,
    Inkling_Small_Recipe,
)
from modal_training_gym.train_recipes.miles_recipe.qwen3_5_4b import (
    Qwen3_5_4B_Miles_Recipe,
)
from modal_training_gym.train_recipes.slime_recipe.qwen3_4b import Qwen3_4B_Recipe
from modal_training_gym.train_recipes.slime_recipe.qwen3_5_4b import Qwen3_5_4B_Recipe
from modal_training_gym.train_recipes.slime_recipe.qwen3_6_27b import Qwen3_6_27B_Recipe
from modal_training_gym.train_recipes.slime_recipe.qwen3_6_35b import Qwen3_6_35B_Recipe
from modal_training_gym.train_recipes.slime_recipe.qwen3_8_27b import Qwen3_8_27B_Recipe
from modal_training_gym.train_recipes.slime_recipe.qwen3_asr_1_7b import (
    Qwen3_ASR_1_7B_Recipe,
)
from modal_training_gym.train_recipes.slime_recipe.qwen3_vl_8b import Qwen3_VL_8B_Recipe

_QWEN35_VL_PROVIDER = "slime_plugins.models.qwen3_5_vl.provide_qwen3_5_vl"


def _mm(modality):
    return MultimodalDataset(
        rows=[{"prompt": "p", "media": ["ref"], "label": "l"}],
        modality=modality,
    )


def _flags(args):
    return {
        args[i]: args[i + 1] for i in range(len(args) - 1) if args[i].startswith("--")
    }


@pytest.mark.parametrize("modality", ["image", "audio"])
def test_multimodal_keys_emitted(modality):
    ds = _mm(modality)
    assert ds.multimodal_keys == {modality: f"{modality}s"}
    if modality == "image":
        flags = _flags(Qwen3_VL_8B_Recipe().cli_args(dataset=ds, model=Qwen3_VL_8B()))
    else:
        flags = _flags(
            Qwen3_ASR_1_7B_Recipe().cli_args(dataset=ds, model=Qwen3_ASR_1_7B())
        )
    assert json.loads(flags["--multimodal-keys"]) == {modality: f"{modality}s"}
    assert flags["--input-key"] == "prompt"
    assert flags["--label-key"] == "label"
    assert ds.apply_chat_template() is (modality == "image")


def test_multimodal_dataset_can_disable_chat_template():
    class ImageDataset(MultimodalDataset):
        def apply_chat_template(self) -> bool:
            return False

    assert ImageDataset(rows=[], modality="image").apply_chat_template() is False


def test_write_writes_media_column(tmp_path):
    ds = MultimodalDataset(
        rows=[{"prompt": "p", "media": ["a.wav", "b.wav"], "label": "l"}],
        modality="audio",
    )
    out = str(tmp_path / "train.jsonl")
    ds.write(out)
    ds.validate_written(out)
    row = json.loads(open(out).readline())
    assert row["audios"] == ["a.wav", "b.wav"]
    assert row["prompt"] == "p" and row["label"] == "l"


def test_huggingface_dataset_rejects_multimodal_keys():
    with pytest.raises(TrainingGymConfigError, match="MultimodalDataset"):
        HuggingFaceDataset(
            hf_repo="statworx/haiku",
            input_column="keywords",
            output_column="text",
            multimodal_keys={"image": "images"},
        )


def test_load_returns_file_paths():
    png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    ds = MultimodalDataset(
        rows=[{"prompt": "p", "media": [f"data:image/png;base64,{png}"], "label": "l"}],
        modality="image",
    )
    path = Path(ds.load()[0]["images"][0])
    assert path.is_file()
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_text_dataset_unaffected():
    ds = HuggingFaceDataset(
        hf_repo="statworx/haiku",
        input_column="keywords",
        output_column="text",
        input_format="text",
    )
    assert ds.multimodal_keys is None
    assert "--multimodal-keys" not in Qwen3_4B_Recipe().cli_args(
        dataset=ds, model=Qwen3_4B()
    )


@pytest.mark.parametrize(
    ("input_format", "input_key", "label_key", "apply_chat_template"),
    [
        ("text", "messages", "label", True),
        ("messages", "prompt", "answer", True),
        ("raw", "prompt", "answer", False),
    ],
)
def test_hugging_face_input_format_controls_dataset_fields(
    input_format, input_key, label_key, apply_chat_template
):
    ds = HuggingFaceDataset(
        hf_repo="some/dataset",
        input_column="prompt",
        output_column="answer",
        input_format=input_format,
    )
    assert ds.input_key() == input_key
    assert ds.label_key() == label_key
    assert ds.apply_chat_template() is apply_chat_template


def test_hugging_face_text_is_formatted_but_messages_pass_through(monkeypatch):
    from datasets import Dataset

    plain_text = Dataset.from_list([{"prompt": "hello", "answer": "world"}])
    monkeypatch.setattr("datasets.load_dataset", lambda *args, **kwargs: plain_text)
    text_dataset = HuggingFaceDataset(
        hf_repo="some/dataset",
        input_column="prompt",
        output_column="answer",
        input_format="text",
    )
    assert list(text_dataset.rows()) == [
        {
            "messages": [{"role": "user", "content": "hello"}],
            "label": "world",
        }
    ]

    messages = [{"role": "user", "content": "hello"}]
    preformatted = Dataset.from_list([{"prompt": messages, "label": "world"}])
    monkeypatch.setattr("datasets.load_dataset", lambda *args, **kwargs: preformatted)
    messages_dataset = HuggingFaceDataset(
        hf_repo="some/dataset",
        input_column="prompt",
        output_column="label",
        input_format="messages",
    )
    assert list(messages_dataset.rows()) == [{"prompt": messages, "label": "world"}]


def test_hugging_face_rejects_unknown_input_format():
    with pytest.raises(ValueError, match="input_format"):
        HuggingFaceDataset(
            hf_repo="some/dataset",
            input_column="prompt",
            output_column="answer",
            input_format="unknown",
        )


def test_media_column_must_be_distinct():
    with pytest.raises(TrainingGymConfigError, match="media_column"):
        MultimodalDataset(rows=[], modality="image", media_column="prompt")


def test_multimodal_dataset_rejects_video():
    with pytest.raises(TrainingGymConfigError, match="image/audio"):
        MultimodalDataset(
            rows=[{"prompt": "p", "media": ["ref"], "label": "l"}],
            modality="video",
        )


def test_train_config_rejects_video():
    ds = _mm("image")
    ds.multimodal_keys = {"video": "videos"}
    with pytest.raises(ValidationError, match="cannot serve video"):
        TrainConfig(
            dataset=ds,
            model=Qwen3_VL_8B(),
            recipe=Qwen3_VL_8B_Recipe(),
        )


@pytest.mark.parametrize("modality", ["audio", "image"])
def test_train_config_rejects_media_on_text_model(modality):
    with pytest.raises(ValidationError, match=f"cannot serve {modality}"):
        TrainConfig(
            dataset=_mm(modality),
            model=Qwen3_4B(),
            recipe=Qwen3_4B_Recipe(),
        )


def test_train_config_accepts_qwen35_image():
    TrainConfig(
        dataset=_mm("image"),
        model=Qwen3_5_4B(),
        recipe=Qwen3_5_4B_Recipe(),
    )


def test_train_config_accepts_asr_audio():
    TrainConfig(
        dataset=_mm("audio"),
        model=Qwen3_ASR_1_7B(),
        recipe=Qwen3_ASR_1_7B_Recipe(),
    )


def test_train_config_rejects_inkling_text_mode_with_audio():
    with pytest.raises(ValidationError, match="cannot serve audio"):
        TrainConfig(
            dataset=_mm("audio"),
            model=Inkling_Small(),
            recipe=Inkling_Small_Recipe(),
        )


def test_train_config_accepts_inkling_audio():
    TrainConfig(
        dataset=_mm("audio"),
        model=Inkling_Small(),
        recipe=Inkling_Small_Recipe(modality="audio"),
    )


def test_qwen35_image_cli_emits_vl_provider(tmp_path):
    recipe = Qwen3_5_4B_Recipe()
    model = Qwen3_5_4B()
    ds = _mm("image")
    args = recipe.cli_args(dataset=ds, model=model)
    flags = _flags(args)
    assert flags["--qkv-format"] == "thd"
    assert flags["--freeze-params-name-list"] == "visual"
    assert "--use-dynamic-batch-size" in args
    assert "--micro-batch-size" not in args
    assert flags["--custom-model-provider-path"] == _QWEN35_VL_PROVIDER
    assert not (recipe.extra_config or {}).get("custom_model_provider_path")

    object.__setattr__(recipe, "extra_config", {"custom_rm_path": "reward.score"})
    prepare_launch_config(
        recipe, None, str(tmp_path), yaml_config_fields=("extra_config",)
    )
    launched = _flags(recipe.cli_args(dataset=ds, model=model))
    assert launched["--custom-model-provider-path"] == _QWEN35_VL_PROVIDER
    assert launched["--custom-config-path"] == recipe.extra_config


def test_qwen35_image_launch_uses_bridge():
    recipe = Qwen3_5_4B_Recipe()
    _apply_qwen35_vl_image_train(recipe, Qwen3_5_4B(), _mm("image"))
    assert recipe.megatron_to_hf_mode == "bridge"
    text = Qwen3_5_4B_Recipe()
    _apply_qwen35_vl_image_train(text, Qwen3_5_4B(), None)
    assert text.megatron_to_hf_mode == ""


def test_qwen35_image_path_extra_config_fails_fast(tmp_path):
    recipe = Qwen3_5_4B_Recipe()
    object.__setattr__(recipe, "extra_config", "configs/extra.yaml")
    with pytest.raises(TrainingGymConfigError, match="custom_model_provider_path"):
        recipe.cli_args(dataset=_mm("image"), model=Qwen3_5_4B())
    launch = Qwen3_5_4B_Recipe()
    object.__setattr__(launch, "extra_config", "configs/extra.yaml")
    with pytest.raises(TrainingGymConfigError, match="custom_model_provider_path"):
        _apply_qwen35_vl_image_train(launch, Qwen3_5_4B(), _mm("image"))

    materialized = Qwen3_5_4B_Recipe()
    object.__setattr__(materialized, "extra_config", {"custom_rm_path": "reward.score"})
    prepare_launch_config(
        materialized,
        None,
        str(tmp_path),
        yaml_config_fields=("extra_config",),
    )
    flags = _flags(materialized.cli_args(dataset=_mm("image"), model=Qwen3_5_4B()))
    assert flags["--custom-model-provider-path"] == _QWEN35_VL_PROVIDER

    owned = Qwen3_5_4B_Recipe()
    object.__setattr__(
        owned,
        "extra_config",
        {"custom_model_provider_path": "slime_plugins.models.other.provide"},
    )
    owned_dir = tmp_path / "owned"
    owned_dir.mkdir()
    prepare_launch_config(
        owned, None, str(owned_dir), yaml_config_fields=("extra_config",)
    )
    owned_flags = _flags(owned.cli_args(dataset=_mm("image"), model=Qwen3_5_4B()))
    assert "--custom-model-provider-path" not in owned_flags
    assert owned_flags["--custom-config-path"] == owned.extra_config


def test_qwen35_recipe_reuse_does_not_stick_vl_provider():
    recipe = Qwen3_5_4B_Recipe()
    image_args = recipe.cli_args(dataset=_mm("image"), model=Qwen3_5_4B())
    text_args = recipe.cli_args(model=Qwen3_5_4B())
    assert _flags(image_args)["--custom-model-provider-path"] == _QWEN35_VL_PROVIDER
    assert "--custom-model-provider-path" not in text_args
    assert not (recipe.extra_config or {}).get("custom_model_provider_path")


def test_qwen35_text_cli_leaves_vl_provider_unset():
    recipe = Qwen3_5_4B_Recipe()
    args = recipe.cli_args(model=Qwen3_5_4B())
    assert "--custom-model-provider-path" not in args
    assert not (recipe.extra_config or {}).get("custom_model_provider_path")


def test_write_jsonl_materializes_data_uris(tmp_path):
    png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    wav = "UklGRiQAAABXQVZFZm10IBAAAAABAAEAESsAACJWAAACABAAZGF0YQAAAAA="
    ds = MultimodalDataset(
        rows=[
            {
                "prompt": "p",
                "media": [f"data:image/png;base64,{png}", "/already/a/path.png"],
                "label": "l",
            }
        ],
        modality="image",
    )
    path = tmp_path / "train.jsonl"
    ds._write_jsonl(ds.load(), str(path))
    row = json.loads(path.read_text().splitlines()[0])
    written, leftover = row["images"]
    assert leftover == "/already/a/path.png"
    assert written == str((tmp_path / "media" / "000000.png").resolve())
    assert path.with_name("media").joinpath("000000.png").read_bytes()[:8] == (
        b"\x89PNG\r\n\x1a\n"
    )

    audio = MultimodalDataset(
        rows=[{"prompt": "p", "media": [f"data:audio/wav;base64,{wav}"], "label": "l"}],
        modality="audio",
    )
    audio_path = tmp_path / "audio.jsonl"
    audio._write_jsonl(audio.load(), str(audio_path))
    audio_row = json.loads(audio_path.read_text().splitlines()[0])
    assert audio_row["audios"] == [str((tmp_path / "media" / "000000.wav").resolve())]
    assert (tmp_path / "media" / "000000.wav").read_bytes()[:4] == b"RIFF"


@pytest.mark.parametrize(
    ("model", "recipe_cls"),
    [
        (Qwen3_6_27B(), Qwen3_6_27B_Recipe),
        (Qwen3_6_35B(), Qwen3_6_35B_Recipe),
        (Qwen3_8_27B(), Qwen3_8_27B_Recipe),
    ],
)
def test_qwen36_38_accept_image(model, recipe_cls):
    TrainConfig(
        dataset=_mm("image"),
        model=model,
        recipe=recipe_cls(),
    )
    recipe = recipe_cls()
    args = recipe.cli_args(dataset=_mm("image"), model=model)
    flags = _flags(args)
    assert flags["--qkv-format"] == "thd"
    assert flags["--freeze-params-name-list"] == "visual"
    assert "--use-dynamic-batch-size" in args
    assert "--micro-batch-size" not in args
    assert flags["--custom-model-provider-path"] == _QWEN35_VL_PROVIDER
    assert not (recipe.extra_config or {}).get("custom_model_provider_path")


def test_miles_qwen35_rejects_image():
    with pytest.raises(ValidationError, match="cannot serve image"):
        TrainConfig(
            dataset=_mm("image"),
            model=Qwen3_5_4B(),
            recipe=Qwen3_5_4B_Miles_Recipe(),
        )


def test_vl_always_emits_bshd():
    flags = _flags(Qwen3_VL_8B_Recipe().cli_args(model=Qwen3_VL_8B()))
    assert flags["--qkv-format"] == "bshd"
    assert flags["--freeze-params-name-list"] == "vision_model"


def test_asr_audio_cli_uses_bshd():
    args = Qwen3_ASR_1_7B_Recipe().cli_args(
        dataset=_mm("audio"), model=Qwen3_ASR_1_7B()
    )
    flags = _flags(args)
    assert flags["--qkv-format"] == "bshd"
    assert "--use-dynamic-batch-size" not in args
    assert flags["--micro-batch-size"] == "1"
    assert "--apply-chat-template" not in args


def test_gemma_image_cli_uses_bshd():
    recipe = Gemma4_26B_A4B_Recipe(modality="vision", rm_type="gemma_math")
    args = recipe.cli_args(dataset=_mm("image"), model=Gemma4_26B_A4B())
    flags = _flags(args)
    assert flags["--qkv-format"] == "bshd"
    assert "--use-dynamic-batch-size" not in args
    assert flags["--micro-batch-size"] == "1"
    assert "--sglang-enable-multimodal" in args


def test_inkling_vision_omits_chat_template():
    recipe = Inkling_Small_Recipe(modality="vision")
    args = recipe.cli_args(dataset=_mm("image"), model=Inkling_Small())
    assert "--apply-chat-template" not in args
    flags = _flags(args)
    assert (
        flags["--custom-model-provider-path"]
        == "miles_plugins.models.inkling.model.inkling_mm_model_provider"
    )


@pytest.mark.parametrize(
    ("modality", "recipe_modality"),
    [("image", "vision"), ("audio", "audio")],
)
def test_inkling_lora_media_keeps_thd_and_dynamic_packing(modality, recipe_modality):
    recipe = Inkling_Small_LoRA_Recipe(modality=recipe_modality)
    args = recipe.cli_args(dataset=_mm(modality), model=Inkling_Small())
    flags = _flags(args)
    assert flags["--qkv-format"] == "thd"
    assert "--use-dynamic-batch-size" in args
    assert "--micro-batch-size" not in args
