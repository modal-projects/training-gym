import json
from pathlib import Path

from modal_training_gym import HuggingFaceDataset, Qwen3_5_4B, TrainConfig
from modal_training_gym.common.dataset import HarborDataset
from modal_training_gym.common.environments.bfcl import BfclMultiTurnDataset
from modal_training_gym.train_recipes.base import BaseTrainRecipe
from modal_training_gym.train_recipes.slime_recipe.qwen3_4b import Qwen3_4b_Recipe
from modal_training_gym.train_recipes.slime_recipe.qwen3_5_4b import Qwen3_5_4b_Recipe


def _flags(args: list[str]) -> dict[str, str]:
    return {
        args[i]: args[i + 1] for i in range(len(args) - 1) if args[i].startswith("--")
    }


class TinyGsm8kDataset(HuggingFaceDataset):
    hf_repo = "example/gsm8k"
    input_column = "question"
    output_column = "answer"
    output_format = "jsonl"

    def rows(self):
        yield {
            self.input_key: [{"role": "user", "content": "1+1?"}],
            self.label_key: "2",
        }


def test_hf_dataset_emits_eval_prompt_data():
    ds = TinyGsm8kDataset()
    args = Qwen3_4b_Recipe().cli_args(dataset=ds)
    flags = _flags(args)

    assert "--eval-prompt-data" in args
    idx = args.index("--eval-prompt-data")
    assert args[idx + 1 : idx + 3] == [
        "eval",
        BaseTrainRecipe._resolve_data_path(ds, "eval"),
    ]

    train_path = flags["--prompt-data"]
    eval_path = args[idx + 2]
    assert "train" in train_path
    assert "eval" in eval_path
    assert train_path != eval_path


def test_model_recipe_can_disable_eval_after_merge():
    class NoEvalTiny(TinyGsm8kDataset):
        writes_eval_paths = False

    ds = NoEvalTiny()
    config = TrainConfig(
        dataset=ds,
        model=Qwen3_5_4B(),
        recipe=Qwen3_5_4b_Recipe(eval_interval=None),
    )
    recipe = config._prepare_recipe()
    args = recipe.cli_args(dataset=ds)

    assert recipe.eval_interval is None
    assert "--eval-interval" not in args
    assert "--eval-prompt-data" not in args


def test_bfcl_dataset_omits_eval_prompt_data():
    ds = BfclMultiTurnDataset()
    args = Qwen3_4b_Recipe(eval_interval=None).cli_args(dataset=ds)
    assert "--eval-interval" not in args
    assert "--eval-prompt-data" not in args


def test_writes_eval_paths_false_skips_eval_file(tmp_path: Path):
    from modal_training_gym.common import launcher_helpers

    class NoEvalTiny(TinyGsm8kDataset):
        writes_eval_paths = False

    ds = NoEvalTiny()
    ds.output_format = "jsonl"
    train_path = str(tmp_path / "train.jsonl")
    eval_path = str(tmp_path / "eval.jsonl")

    def resolve(_dataset, split: str) -> str:
        return train_path if split == "train" else eval_path

    launcher_helpers.materialize_dataset(ds, resolve)

    assert Path(train_path).exists()
    assert not Path(eval_path).exists()


def test_harbor_materializes_eval_split_rows(tmp_path: Path):
    from modal_training_gym.common import launcher_helpers

    tasks = tmp_path / "tasks"
    for name in ("task_a", "task_b"):
        task_dir = tasks / name
        task_dir.mkdir(parents=True)
        (task_dir / "instruction.md").write_text(f"Do {name}", encoding="utf-8")

    ds = HarborDataset(
        task_root=str(tasks),
        train_size=1,
        eval_size=1,
        shuffle_tasks=False,
        split="train",
    )
    ds.output_format = "jsonl"
    train_path = str(tmp_path / "train.jsonl")
    eval_path = str(tmp_path / "eval.jsonl")

    def resolve(_dataset, split: str) -> str:
        return train_path if split == "train" else eval_path

    launcher_helpers.materialize_dataset(ds, resolve)

    train_row = json.loads(Path(train_path).read_text())
    eval_row = json.loads(Path(eval_path).read_text())
    assert train_row != eval_row
    assert train_path != eval_path
