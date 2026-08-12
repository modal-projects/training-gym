import zipfile
from pathlib import Path

import pytest

from scripts.partition_harbor_dataset import (
    DEFAULT_DATA_VOLUME,
    ArchivedHarborSource,
    mixed_subset_name,
    read_jsonl,
    repo_disjoint_split,
    sha256,
    write_mixed_subset,
    write_partitions,
)


def test_partition_and_recipe_share_slime_data_volume() -> None:
    assert DEFAULT_DATA_VOLUME == "slime-data"


def _rows(groups: int = 650) -> list[dict]:
    rows = []
    for group_index in range(groups):
        language = "python" if group_index % 2 == 0 else "typescript"
        for task_index in range(2):
            rows.append(
                {
                    "prompt": f"task {group_index}-{task_index}",
                    "label": "{}",
                    "metadata": {
                        "instance_id": f"{group_index}-{task_index}",
                        "task_path": f"fixture/tasks/repo{group_index}__{task_index}",
                        "source": {"language_bucket": language},
                    },
                }
            )
    return rows


def test_partitions_use_fixed_names_nested_rows_and_golden_hashes(
    tmp_path: Path,
) -> None:
    counts = write_partitions(tmp_path, _rows())

    assert counts == {
        "eval": 260,
        "train-full": 1040,
        "train-100": 100,
        "train-300": 300,
        "train-1000": 1000,
    }
    assert {path.name for path in tmp_path.glob("*.jsonl")} == {
        "eval.jsonl",
        "train-full.jsonl",
        "train-100.jsonl",
        "train-300.jsonl",
        "train-1000.jsonl",
    }
    ids = {
        size: {
            row["metadata"]["instance_id"]
            for row in read_jsonl(tmp_path / f"train-{size}.jsonl")
        }
        for size in (100, 300, 1000)
    }
    assert ids[100] < ids[300] < ids[1000]

    repeated = tmp_path / "repeated"
    write_partitions(repeated, _rows())
    assert (tmp_path / "eval.jsonl").read_bytes() == (
        repeated / "eval.jsonl"
    ).read_bytes()
    assert (tmp_path / "train-300.jsonl").read_bytes() == (
        repeated / "train-300.jsonl"
    ).read_bytes()
    assert sha256(tmp_path / "eval.jsonl") == (
        "733fbc13b0c570e81b12e445d6e80385405767ddde7a84bb838cbdf8b35a6d20"
    )
    assert sha256(tmp_path / "train-300.jsonl") == (
        "d025f37455ada1142fd8c2db85ea2f74719ed6499388e7e51f3384a33768bde2"
    )


def test_eval_partition_is_repository_disjoint() -> None:
    train_rows, eval_rows = repo_disjoint_split(
        _rows(8),
        eval_fraction=0.25,
        seed=7,
        metadata_namespace="source",
    )

    def repos(rows: list[dict]) -> set[str]:
        return {
            Path(row["metadata"]["task_path"]).name.rsplit("__", 1)[0] for row in rows
        }

    assert repos(train_rows).isdisjoint(repos(eval_rows))
    assert len(train_rows) + len(eval_rows) == 16


def test_mixed_subset_uses_recipe_and_sample_count_without_profile(
    tmp_path: Path,
) -> None:
    write_partitions(tmp_path, _rows())
    source_rows = read_jsonl(tmp_path / "train-100.jsonl")
    instance_ids = [row["metadata"]["instance_id"] for row in source_rows[:3]]
    samples = []
    for instance_id, solved, episodes in zip(
        instance_ids,
        (2, 4, 1),
        (4, 4, 3),
    ):
        for index in range(episodes):
            samples.append(
                {
                    "index": index,
                    "metadata": {
                        "instance_id": instance_id,
                        "agentic": {"is_solved": index < solved},
                    },
                }
            )

    output, provenance = write_mixed_subset(
        tmp_path,
        source="train-100",
        recipe="Qwen3_6_27b_Agentic_Recipe",
        samples=samples,
        n_samples=4,
        checkpoint="base",
        probe_dump="/checkpoints/probe.pt",
    )

    assert output.name == ("train-100-mixed-reward-qwen3-6-27b-agentic-recipe-n4.jsonl")
    assert provenance["recipe"] == "Qwen3_6_27b_Agentic_Recipe"
    assert len(provenance["selected_instance_ids"]) == 1
    assert read_jsonl(output)[0]["metadata"]["instance_id"] == instance_ids[0]
    assert output.with_suffix(".json").is_file()
    with pytest.raises(FileExistsError):
        write_mixed_subset(
            tmp_path,
            source="train-100",
            recipe="Qwen3_6_27b_Agentic_Recipe",
            samples=samples,
            n_samples=4,
            checkpoint="base",
            probe_dump="/checkpoints/probe.pt",
        )


def test_mixed_subset_name_rejects_empty_recipe() -> None:
    with pytest.raises(ValueError, match="letter or number"):
        mixed_subset_name("train-300", "---", 8)


def test_archive_extraction_rejects_path_traversal(tmp_path: Path) -> None:
    tasks = tmp_path / "tasks"
    tasks.mkdir()
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("../escape/task.toml", "")

    with pytest.raises(ValueError, match="unsafe path"):
        ArchivedHarborSource.safe_extract(archive, tasks)
