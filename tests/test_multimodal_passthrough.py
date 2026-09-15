"""The multimodal passthrough: a dataset names its media column, the recipe
forwards it to slime as --multimodal-keys. Modality-agnostic (image/audio).
"""

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
    ModelConfig,
    QWEN3_5_VL_PROVIDER,
    Qwen3_4B,
    Qwen3_5_0_8B,
    Qwen3_5_2B,
    Qwen3_5_4B,
    Qwen3_5_9B,
    Qwen3_6_27B,
    Qwen3_6_35B,
    Qwen3_8_27B,
    Qwen3_ASR_1_7B,
    Qwen3_VL_8B,
)
from modal_training_gym.common.train import TrainConfig
from modal_training_gym.frameworks.slime.launcher import build_slime_app
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
from modal_training_gym.train_recipes.slime_recipe.recipe import SlimeRecipe
from modal_training_gym.train_recipes.slime_recipe.qwen3_asr_1_7b import (
    Qwen3_ASR_1_7B_Recipe,
)
from modal_training_gym.train_recipes.slime_recipe.qwen3_vl_8b import Qwen3_VL_8B_Recipe


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
    assert ds.apply_chat_template() is True


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
    with pytest.raises(TrainingGymConfigError, match="image/audio"):
        MultimodalDataset(
            rows=[{"prompt": "p", "media": ["ref"], "label": "l"}],
            modality="video",
        )


def test_train_config_rejects_unknown_multimodal_key():
    ds = _mm("image")
    ds.multimodal_keys = {"pictures": "col"}
    with pytest.raises(ValidationError, match=r"pictures.*allowed: audio, image"):
        TrainConfig(
            dataset=ds,
            model=Qwen3_VL_8B(),
            recipe=Qwen3_VL_8B_Recipe(),
        )


def test_train_config_rejects_media_on_text_model():
    with pytest.raises(ValidationError, match="cannot serve image"):
        TrainConfig(
            dataset=_mm("image"),
            model=Qwen3_4B(),
            recipe=Qwen3_4B_Recipe(),
        )


def test_recipe_active_modalities():
    with pytest.raises(ValidationError, match="cannot serve audio"):
        TrainConfig(
            dataset=_mm("audio"),
            model=Inkling_Small(),
            recipe=Inkling_Small_Recipe(),
        )
    TrainConfig(
        dataset=_mm("audio"),
        model=Inkling_Small(),
        recipe=Inkling_Small_Recipe(modality="audio"),
    )
    TrainConfig(
        dataset=_mm("image"),
        model=Inkling_Small(),
        recipe=Inkling_Small_Recipe(modality="vision"),
    )
    with pytest.raises(
        ValidationError, match="Inkling_Small_LoRA_Recipe does not support audio"
    ):
        Inkling_Small_LoRA_Recipe(modality="audio")
    with pytest.raises(
        ValidationError, match="Qwen3_5_4B_Miles_Recipe cannot serve image"
    ):
        TrainConfig(
            dataset=_mm("image"),
            model=Qwen3_5_4B(),
            recipe=Qwen3_5_4B_Miles_Recipe(),
        )


def test_qwen35_image_path_extra_config_fails_fast(tmp_path):
    recipe = Qwen3_5_4B_Recipe()
    object.__setattr__(recipe, "extra_config", "configs/extra.yaml")
    with pytest.raises(TrainingGymConfigError, match="custom_model_provider"):
        recipe.cli_args(dataset=_mm("image"), model=Qwen3_5_4B())

    materialized = Qwen3_5_4B_Recipe()
    object.__setattr__(materialized, "extra_config", {"custom_rm_path": "reward.score"})
    prepare_launch_config(
        materialized,
        None,
        str(tmp_path),
        yaml_config_fields=("extra_config",),
    )
    flags = _flags(materialized.cli_args(dataset=_mm("image"), model=Qwen3_5_4B()))
    assert flags["--custom-model-provider-path"] == QWEN3_5_VL_PROVIDER

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
    assert _flags(image_args)["--custom-model-provider-path"] == QWEN3_5_VL_PROVIDER
    assert "--custom-model-provider-path" not in text_args
    assert "--megatron-to-hf-mode" not in text_args
    assert not (recipe.extra_config or {}).get("custom_model_provider_path")


def test_yaml_bridge_mode_does_not_assign_torch_dist_ref_load():
    recipe = Qwen3_5_4B_Recipe(extra_config={"megatron_to_hf_mode": "bridge"})
    build_slime_app(
        training_run_id="bridge-yaml",
        slime=recipe,
        model=Qwen3_5_4B(),
        dataset=_mm("image"),
    )
    assert recipe.ref_load == ""


def test_yaml_raw_mode_overrides_bridge_field_for_conversion():
    model = Qwen3_5_4B()
    dataset = _mm("image")
    recipe = Qwen3_5_4B_Recipe(
        megatron_to_hf_mode="bridge",
        extra_config={"megatron_to_hf_mode": "raw"},
    )
    assert recipe.megatron_to_hf_mode == "bridge"
    assert recipe.effective_megatron_to_hf_mode(dataset, model) == "raw"
    build_slime_app(
        training_run_id="raw-yaml",
        slime=recipe,
        model=model,
        dataset=dataset,
    )
    assert recipe.ref_load == "/checkpoints/torch_dist/Qwen--Qwen3.5-4B-v31"


def test_write_jsonl_materializes_data_uris(tmp_path):
    png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    wav = "UklGRiQAAABXQVZFZm10IBAAAAABAAEAESsAACJWAAACABAAZGF0YQAAAAA="
    url = "https://example.com/img.png?" + ("a" * 8000)
    ds = MultimodalDataset(
        rows=[
            {
                "prompt": "p",
                "media": [
                    f"data:image/png;base64,{png}",
                    "/already/a/path.png",
                    b"train-bytes",
                    "data:image/png,%89PNG%0D%0A%1A%0A",
                    url,
                ],
                "label": "l",
            }
        ],
        modality="image",
    )
    path = tmp_path / "train.jsonl"
    ds.write(str(path))
    row = json.loads(path.read_text().splitlines()[0])
    images = row["images"]
    media_dir = path.with_name(path.name + ".media")
    assert images[0] == str((media_dir / "000000.png").resolve())
    assert images[1] == "/already/a/path.png"
    assert Path(images[0]).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert Path(images[2]).read_bytes() == b"train-bytes"
    assert Path(images[3]).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert images[4] == url

    audio = MultimodalDataset(
        rows=[{"prompt": "p", "media": [f"data:audio/wav;base64,{wav}"], "label": "l"}],
        modality="audio",
    )
    audio_path = tmp_path / "audio.jsonl"
    audio.write(str(audio_path))
    audio_row = json.loads(audio_path.read_text().splitlines()[0])
    audio_media = audio_path.with_name(audio_path.name + ".media")
    assert audio_row["audios"] == [str((audio_media / "000000.wav").resolve())]
    assert (audio_media / "000000.wav").read_bytes()[:4] == b"RIFF"

    eval_ds = MultimodalDataset(
        rows=[{"prompt": "p", "media": [b"eval-bytes"], "label": "l"}],
        modality="image",
    )
    eval_path = tmp_path / "eval.jsonl"
    eval_ds.write(str(eval_path))
    train_media = Path(images[2])
    eval_media = Path(json.loads(eval_path.read_text().splitlines()[0])["images"][0])
    assert train_media != eval_media
    assert eval_media.read_bytes() == b"eval-bytes"


def test_train_config_validates_eval_dataset():
    with pytest.raises(ValidationError, match="cannot serve audio"):
        TrainConfig(
            dataset=_mm("image"),
            model=Qwen3_VL_8B(),
            recipe=Qwen3_VL_8B_Recipe(),
            eval_dataset=_mm("audio"),
        )
    eval_ds = MultimodalDataset(
        rows=[{"prompt": "p", "media": ["ref"], "label": "l"}],
        modality="image",
        media_column="pictures",
    )
    with pytest.raises(ValidationError, match="multimodal_keys"):
        TrainConfig(
            dataset=_mm("image"),
            model=Qwen3_VL_8B(),
            recipe=Qwen3_VL_8B_Recipe(),
            eval_dataset=eval_ds,
        )


_QWEN35_LINE_MODELS = (
    Qwen3_5_0_8B,
    Qwen3_5_2B,
    Qwen3_5_4B,
    Qwen3_5_9B,
)

_QWEN36_38_TEXT_MODELS = (Qwen3_6_27B, Qwen3_6_35B, Qwen3_8_27B)


@pytest.mark.parametrize("model_cls", _QWEN35_LINE_MODELS, ids=lambda c: c.__name__)
def test_qwen35_line_accepts_image(model_cls):
    model = model_cls()
    recipe = SlimeRecipe.get_base_recipe(model)
    TrainConfig(dataset=_mm("image"), model=model, recipe=recipe)
    args = recipe.cli_args(dataset=_mm("image"), model=model)
    flags = _flags(args)
    assert flags["--freeze-params-name-list"] == "visual"
    assert flags["--custom-model-provider-path"] == QWEN3_5_VL_PROVIDER
    assert flags["--megatron-to-hf-mode"] == "bridge"
    assert not (recipe.extra_config or {}).get("custom_model_provider_path")


@pytest.mark.parametrize("model_cls", _QWEN36_38_TEXT_MODELS, ids=lambda c: c.__name__)
def test_qwen36_38_reject_image(model_cls):
    model = model_cls()
    recipe = SlimeRecipe.get_base_recipe(model)
    with pytest.raises(ValidationError, match="cannot serve image"):
        TrainConfig(dataset=_mm("image"), model=model, recipe=recipe)
    flags = _flags(recipe.cli_args(dataset=_mm("image"), model=model))
    assert flags.get("--custom-model-provider-path") != QWEN3_5_VL_PROVIDER


def test_qwen35_4b_image_uses_thd():
    model = Qwen3_5_4B()
    recipe = Qwen3_5_4B_Recipe()
    args = recipe.cli_args(dataset=_mm("image"), model=model)
    flags = _flags(args)
    assert flags["--qkv-format"] == "thd"
    assert "--use-dynamic-batch-size" in args
    assert "--micro-batch-size" not in args


def test_qwen3_vl_does_not_use_qwen35_line_provider():
    flags = _flags(
        Qwen3_VL_8B_Recipe().cli_args(dataset=_mm("image"), model=Qwen3_VL_8B())
    )
    assert flags.get("--custom-model-provider-path") != QWEN3_5_VL_PROVIDER


def test_custom_model_provider_forwards_as_path():
    provider = "slime_plugins.models.other.provide"
    provided = SlimeRecipe().overrides(
        _mm("image"),
        ModelConfig(custom_model_provider=provider),
    )
    assert provided["custom_model_provider_path"] == provider


@pytest.mark.parametrize(
    ("recipe", "model", "dataset"),
    [
        (Qwen3_VL_8B_Recipe(), Qwen3_VL_8B(), None),
        (Qwen3_ASR_1_7B_Recipe(), Qwen3_ASR_1_7B(), _mm("audio")),
    ],
    ids=["vl", "asr"],
)
def test_bshd_cli(recipe, model, dataset):
    args = recipe.cli_args(dataset=dataset, model=model)
    flags = _flags(args)
    extra = recipe.extra_config or {}
    assert extra.get("qkv_format") == "bshd"
    assert extra.get("micro_batch_size") == 1
    assert recipe.use_dynamic_batch_size is False
    if isinstance(recipe, Qwen3_VL_8B_Recipe):
        assert flags["--freeze-params-name-list"] == "vision_model"
        return
    assert "--use-dynamic-batch-size" not in args
    assert "--apply-chat-template" not in args


def test_gemma_image_cli_uses_bshd():
    recipe = Gemma4_26B_A4B_Recipe(modality="vision", rm_type="gemma_math")
    args = recipe.cli_args(dataset=_mm("image"), model=Gemma4_26B_A4B())
    flags = _flags(args)
    assert flags["--qkv-format"] == "bshd"
    assert "--use-dynamic-batch-size" not in args
    assert flags["--micro-batch-size"] == "1"
    assert "--sglang-enable-multimodal" in args


def test_inkling_media_omits_chat_template():
    recipe = Inkling_Small_Recipe(modality="vision")
    args = recipe.cli_args(dataset=_mm("image"), model=Inkling_Small())
    assert "--apply-chat-template" not in args
    flags = _flags(args)
    assert (
        flags["--custom-model-provider-path"]
        == "miles_plugins.models.inkling.model.inkling_mm_model_provider"
    )
