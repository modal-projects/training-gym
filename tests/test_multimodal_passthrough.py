"""The multimodal passthrough: a dataset names its media column, the recipe
forwards it to slime as --multimodal-keys. Modality-agnostic (image/audio/video).
"""

import json

import pytest

from modal_training_gym import HuggingFaceDataset, MultimodalDataset, SlimeRecipe
from modal_training_gym.common.launcher_helpers import run_prepare_dataset

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


def test_eval_dataset_uses_separate_materialization_path():
    train = MultimodalDataset(rows=[])
    evaluation = MultimodalDataset(rows=[])

    args = SlimeRecipe(**_RECIPE_KW).cli_args(
        dataset=train,
        eval_dataset=evaluation,
    )
    flags = _flags(args)

    assert flags["--prompt-data"] == "/data/MultimodalDataset/train.jsonl"
    eval_index = args.index("--eval-prompt-data")
    assert args[eval_index + 1 : eval_index + 3] == [
        "eval",
        "/data/MultimodalDataset/eval.jsonl",
    ]


def test_train_and_eval_datasets_are_written_separately(tmp_path):
    train = MultimodalDataset(
        rows=[{"prompt": "train", "media": [], "label": "train-label"}]
    )
    evaluation = MultimodalDataset(
        rows=[{"prompt": "eval", "media": [], "label": "eval-label"}]
    )

    class FakeVolume:
        reloads = 0
        commits = 0

        def reload(self):
            self.reloads += 1

        def commit(self):
            self.commits += 1

    volume = FakeVolume()

    def resolve_path(_dataset, split):
        return str(tmp_path / f"{split}.jsonl")

    run_prepare_dataset(train, evaluation, volume, resolve_path)

    train_row = json.loads((tmp_path / "train.jsonl").read_text().splitlines()[0])
    eval_row = json.loads((tmp_path / "eval.jsonl").read_text().splitlines()[0])
    assert train_row["prompt"] == "train"
    assert eval_row["prompt"] == "eval"
    assert volume.reloads == 1
    assert volume.commits == 1


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
