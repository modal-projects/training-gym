"""The multimodal passthrough: a dataset names its media column, the recipe
forwards it to slime as --multimodal-keys. Modality-agnostic (image/audio/video).
"""

import json

import pytest

from modal_training_gym import HuggingFaceDataset, MultimodalDataset, SlimeRecipe

_RECIPE_KW = dict(
    gpu_type="H100",
    colocate=True,
    tensor_model_parallel_size=1,
    sequence_parallel=False,
    rollout_num_gpus_per_engine=1,
    num_rollout=1,
    rollout_batch_size=4,
    rollout_max_response_len=256,
    rollout_temperature=1.0,
    save_interval=1,
)


def _flags(args):
    return {
        args[i]: args[i + 1] for i in range(len(args) - 1) if args[i].startswith("--")
    }


@pytest.mark.parametrize("modality", ["image", "audio", "video"])
def test_multimodal_keys_emitted(modality):
    class TestDataset(MultimodalDataset):
        pass

    TestDataset.modality = modality
    rows = [{"prompt": "p", "media": ["ref"], "label": "l"}]
    ds = TestDataset(rows=rows)
    assert ds.multimodal_keys == {modality: f"{modality}s"}

    flags = _flags(SlimeRecipe(**_RECIPE_KW).cli_args(dataset=ds))
    assert json.loads(flags["--multimodal-keys"]) == {modality: f"{modality}s"}
    assert flags["--input-key"] == "prompt"
    assert flags["--label-key"] == "label"


def test_write_writes_media_column(tmp_path):
    rows = [{"prompt": "p", "media": ["a.wav", "b.wav"], "label": "l"}]
    ds = MultimodalDataset(rows=rows)
    out = str(tmp_path / "train.jsonl")
    ds.write(out)
    ds.validate_write(out)  # must not raise
    row = json.loads(open(out).readline())
    assert row["audios"] == ["a.wav", "b.wav"]
    assert row["prompt"] == "p" and row["label"] == "l"


def test_dataset_does_not_add_eval_prompt_data():
    ds = MultimodalDataset(rows=[])
    args = SlimeRecipe(**_RECIPE_KW).cli_args(dataset=ds)

    assert "--eval-prompt-data" not in args


def test_huggingface_write_uses_rows_override(tmp_path):
    class TestDataset(HuggingFaceDataset):
        hf_repo = "unused"
        output_format = "jsonl"

        def rows(self):
            return [{"input": "custom", "label": "answer"}]

    output_path = tmp_path / "dataset.jsonl"
    TestDataset().write(str(output_path))

    assert json.loads(output_path.read_text())["input"] == "custom"


def test_text_dataset_unaffected():
    class TextDataset(HuggingFaceDataset):
        hf_repo = "statworx/haiku"
        input_column = "keywords"
        output_column = "text"

    ds = TextDataset()
    assert getattr(ds, "multimodal_keys", None) is None
    assert "--multimodal-keys" not in SlimeRecipe(**_RECIPE_KW).cli_args(dataset=ds)


def test_media_column_must_be_distinct():
    class InvalidDataset(MultimodalDataset):
        modality = "image"
        media_column = "prompt"

    with pytest.raises(ValueError):
        InvalidDataset(rows=[])
