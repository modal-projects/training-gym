import json
import sys
import types
from pathlib import Path

import pytest

from modal_training_gym.train_recipes.slime_recipe import Qwen3_6_27B_Recipe_Agentic
from scripts.partition_swe_dataset import (
    DEFAULT_SWE_DATASET,
    SWE_DATASETS,
    SweBenchSource,
    data_volume_name,
    dataset_root_name,
    mixed_subset_name,
    read_jsonl,
    repo_disjoint_split,
    sha256,
    write_jsonl,
    write_mixed_subset,
    write_partitions,
)


def test_script_writes_to_the_volume_the_agentic_recipe_mounts() -> None:
    recipe = Qwen3_6_27B_Recipe_Agentic()
    assert recipe.data_volume_name, "the recipe must name a shared data volume"
    assert data_volume_name(recipe) == recipe.data_volume_name


def test_dataset_keys_are_valid_dataset_roots() -> None:
    assert DEFAULT_SWE_DATASET in SWE_DATASETS
    for dataset in SWE_DATASETS.values():
        assert dataset_root_name(dataset.key) == dataset.key


def _rows(groups: int = 650) -> list[dict]:
    rows = []
    for group_index in range(groups):
        language = "python" if group_index % 2 == 0 else "ts"
        for task_index in range(2):
            rows.append(
                {
                    "prompt": f"task {group_index}-{task_index}",
                    "label": "{}",
                    "metadata": {
                        "instance_id": f"{group_index}-{task_index}",
                        "task_path": f"fixture/tasks/repo{group_index}__{task_index}",
                        "source": {
                            "repo": f"org/repo{group_index}",
                            "language": language,
                        },
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
        "eval-4": 4,
        "eval-100": 100,
        "train-full": 1040,
        "train-4": 4,
        "train-100": 100,
        "train-300": 300,
        "train-1000": 1000,
    }
    assert {path.name for path in tmp_path.glob("*.jsonl")} == {
        f"{name}.jsonl" for name in counts
    }
    for prefix, sizes in (("train", (4, 100, 300, 1000)), ("eval", (4, 100))):
        ids = [
            {
                row["metadata"]["instance_id"]
                for row in read_jsonl(tmp_path / f"{prefix}-{size}.jsonl")
            }
            for size in sizes
        ]
        for smaller, larger in zip(ids, ids[1:]):
            assert smaller < larger

    repeated = tmp_path / "repeated"
    write_partitions(repeated, _rows())
    assert (tmp_path / "eval.jsonl").read_bytes() == (
        repeated / "eval.jsonl"
    ).read_bytes()
    assert (tmp_path / "train-300.jsonl").read_bytes() == (
        repeated / "train-300.jsonl"
    ).read_bytes()
    assert (
        sha256(tmp_path / "eval.jsonl")
        == "5cb9d4ac1edaf5845ace491486d86e4e5603d583efa0f6bbd8459f609542fe78"
    )
    assert (
        sha256(tmp_path / "train-300.jsonl")
        == "593165a635286748fa140e50b3f81db1d74583b85ed5386987ef062c7c8fb3de"
    )


def test_sized_splits_larger_than_their_pool_are_skipped(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    counts = write_partitions(tmp_path, _rows(10))

    assert counts == {"eval": 4, "eval-4": 4, "train-full": 16, "train-4": 4}
    assert "skipping train-100: only 16 train rows" in capsys.readouterr().out


def test_eval_partition_is_task_group_disjoint() -> None:
    train_rows, eval_rows = repo_disjoint_split(
        _rows(8),
        eval_fraction=0.25,
        seed=7,
        metadata_namespace="source",
    )

    def repos(rows: list[dict]) -> set[str]:
        return {row["metadata"]["source"]["repo"] for row in rows}

    assert repos(train_rows).isdisjoint(repos(eval_rows))
    assert len(train_rows) + len(eval_rows) == 16


def _probe_samples(instance_ids: list[str], solved: list[int]) -> list[dict]:
    return [
        {
            "index": index,
            "metadata": {
                "instance_id": instance_id,
                "agentic": {"is_solved": index < solved_count},
            },
        }
        for instance_id, solved_count in zip(instance_ids, solved, strict=True)
        for index in range(4)
    ]


def test_mixed_subset_uses_recipe_and_sample_count_without_profile(
    tmp_path: Path,
) -> None:
    write_partitions(tmp_path, _rows())
    source_rows = read_jsonl(tmp_path / "train-100.jsonl")
    instance_ids = [row["metadata"]["instance_id"] for row in source_rows]
    samples = _probe_samples(instance_ids, solved=[2, 4, 0] + [4] * 97)

    output, provenance = write_mixed_subset(
        tmp_path,
        source="train-100",
        recipe="Qwen3_6_27B_Recipe_Agentic",
        samples=samples,
        n_samples=4,
        checkpoint="base",
        probe_dump="/checkpoints/probe.pt",
    )

    assert output.name == ("train-100-mixed-reward-qwen3-6-27b-recipe-agentic-n4.jsonl")
    assert provenance["recipe"] == "Qwen3_6_27B_Recipe_Agentic"
    assert len(provenance["selected_instance_ids"]) == 1
    assert read_jsonl(output)[0]["metadata"]["instance_id"] == instance_ids[0]
    assert output.with_suffix(".json").is_file()
    with pytest.raises(FileExistsError):
        write_mixed_subset(
            tmp_path,
            source="train-100",
            recipe="Qwen3_6_27B_Recipe_Agentic",
            samples=samples,
            n_samples=4,
            checkpoint="base",
            probe_dump="/checkpoints/probe.pt",
        )


@pytest.mark.parametrize("drop, extra", [(1, 0), (0, 1), (1, 1)])
def test_mixed_subset_rejects_probes_that_do_not_cover_the_source(
    tmp_path: Path, drop: int, extra: int
) -> None:
    write_partitions(tmp_path, _rows())
    instance_ids = [
        row["metadata"]["instance_id"]
        for row in read_jsonl(tmp_path / "train-100.jsonl")
    ]
    probed = instance_ids[drop:] + ["not-in-train-100"] * extra
    with pytest.raises(ValueError, match="does not cover train-100"):
        write_mixed_subset(
            tmp_path,
            source="train-100",
            recipe="Qwen3_6_27B_Recipe_Agentic",
            samples=_probe_samples(probed, solved=[2] * len(probed)),
            n_samples=4,
            checkpoint="base",
            probe_dump="/checkpoints/probe.pt",
        )


def test_mixed_subset_rejects_dumps_without_fixed_samples_per_task(
    tmp_path: Path,
) -> None:
    write_partitions(tmp_path, _rows())
    instance_id = read_jsonl(tmp_path / "train-100.jsonl")[0]["metadata"]["instance_id"]
    samples = [
        {
            "index": 0,
            "metadata": {"instance_id": instance_id, "agentic": {"is_solved": True}},
        }
    ]
    with pytest.raises(ValueError, match="fixed-sample evaluation dump"):
        write_mixed_subset(
            tmp_path,
            source="train-100",
            recipe="Qwen3_6_27B_Recipe_Agentic",
            samples=samples,
            n_samples=4,
            checkpoint="base",
            probe_dump="/checkpoints/probe.pt",
        )


def test_mixed_subset_rejects_non_train_sources(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="source must be one of"):
        write_mixed_subset(
            tmp_path,
            source="eval",
            recipe="r",
            samples=[],
            n_samples=4,
            checkpoint="base",
            probe_dump="/checkpoints/probe.pt",
        )


def test_mixed_subset_name_rejects_empty_recipe() -> None:
    with pytest.raises(ValueError, match="letter or number"):
        mixed_subset_name("train-300", "---", 8)


@pytest.mark.parametrize("value", ["", ".", "..", "org/repo", "..\\outside", "/data"])
def test_dataset_root_must_be_a_single_directory_name(value: str) -> None:
    with pytest.raises(ValueError, match="single directory name"):
        dataset_root_name(value)
    assert dataset_root_name("swe_rebench_v2") == "swe_rebench_v2"


def _source(**overrides) -> SweBenchSource:
    kwargs = dict(
        dataset=SWE_DATASETS[DEFAULT_SWE_DATASET],
        hf_revision=None,
        metadata_namespace="source",
        translator_revision="a" * 40,
        min_grade="A",
        limit=None,
    )
    kwargs.update(overrides)
    return SweBenchSource(**kwargs)


def test_converted_rows_are_reused_only_for_an_identical_source(
    tmp_path: Path,
) -> None:
    source = _source()
    record = source.source_record(revision="deadbeef")
    assert record["hf_repo"] == "nebius/SWE-rebench-V2"
    assert SweBenchSource.cached_rows(tmp_path, record) is None

    rows = _rows(1)
    write_jsonl(tmp_path / "all.converted.jsonl", rows)
    (tmp_path / "all.converted.json").write_text(json.dumps(record))
    assert SweBenchSource.cached_rows(tmp_path, record) == rows

    for changed in (
        {**record, "revision": "cafebabe"},
        {**record, "split": "other"},
        {**record, "translator_revision": "b" * 40},
        {**record, "min_grade": None},
        {**record, "limit": 10},
    ):
        assert SweBenchSource.cached_rows(tmp_path, changed) is None


def test_source_refresh_drops_converted_tasks_and_derived_mixed_subsets(
    tmp_path: Path,
) -> None:
    (tmp_path / "tasks" / "repo__1").mkdir(parents=True)
    write_partitions(tmp_path, _rows())
    stale = tmp_path / "train-100-mixed-reward-qwen3-6-27b-agentic-n8"
    stale.with_suffix(".jsonl").write_text("")
    stale.with_suffix(".json").write_text("{}")

    SweBenchSource.clear_converted(tmp_path)

    assert not (tmp_path / "tasks").exists()
    assert not list(tmp_path.glob("*-mixed-reward-*"))
    assert (tmp_path / "train-100.jsonl").is_file()


def _install_fake_fork(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stand in for the fork's converters, which only exist inside the training image."""
    swerebench = types.ModuleType("agentic_rl.environment.convert2slime.swerebench")
    harbor = types.ModuleType("agentic_rl.environment.convert2slime.harbor")

    class SkipRow(Exception):
        pass

    class SkipTask(Exception):
        pass

    def build_task_dir(row: dict, dest: Path) -> None:
        if row["language"] != "python":
            raise SkipRow(f"unsupported language {row['language']!r}")
        dest.mkdir(parents=True)
        (dest / "task.toml").write_text("")

    def translate_task(task_dir: Path, *, dataset: str) -> dict:
        if task_dir.name.endswith("gpu"):
            raise SkipTask("requires GPU/TPU")
        return {
            "prompt": [{"role": "user", "content": "fix it"}],
            "label": task_dir.name,
            "metadata": {"instance_id": task_dir.name, "dataset": dataset},
        }

    swerebench.SkipRow = SkipRow  # type: ignore[attr-defined]
    swerebench._passes_quality = lambda row, min_grade: (
        row.get("grade", "A")
        <= (  # type: ignore[attr-defined]
            min_grade or "Z"
        )
    )
    swerebench._safe_id = lambda instance_id: instance_id.replace("/", "__")  # type: ignore[attr-defined]
    swerebench.build_task_dir = build_task_dir  # type: ignore[attr-defined]
    harbor.SkipTask = SkipTask  # type: ignore[attr-defined]
    harbor.translate_task = translate_task  # type: ignore[attr-defined]

    package = types.ModuleType("agentic_rl.environment.convert2slime")
    package.swerebench = swerebench  # type: ignore[attr-defined]
    package.harbor = harbor  # type: ignore[attr-defined]
    for name, module in (
        ("agentic_rl", types.ModuleType("agentic_rl")),
        ("agentic_rl.environment", types.ModuleType("agentic_rl.environment")),
        ("agentic_rl.environment.convert2slime", package),
        ("agentic_rl.environment.convert2slime.swerebench", swerebench),
        ("agentic_rl.environment.convert2slime.harbor", harbor),
    ):
        monkeypatch.setitem(sys.modules, name, module)


def test_convert_offers_every_row_and_counts_what_the_fork_skips(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    _install_fake_fork(monkeypatch)
    hub_rows = [
        {"instance_id": "a/b-1", "repo": "a/b", "language": "python", "license": "MIT"},
        {"instance_id": "a/b-2", "repo": "a/b", "language": "go"},
        {"instance_id": "c/d-1", "repo": "c/d", "language": "python", "grade": "C"},
        {"instance_id": "c/d-gpu", "repo": "c/d", "language": "python"},
        {"instance_id": "e/f-1", "repo": "e/f", "language": "python"},
    ]
    source = _source()
    monkeypatch.setattr(source, "rows", lambda revision: iter(hub_rows))
    root = tmp_path / "swe_rebench_v2"
    root.mkdir()

    rows = source.convert(root, revision="deadbeef")

    assert [row["metadata"]["instance_id"] for row in rows] == ["a__b-1", "e__f-1"]
    assert rows[0]["metadata"]["task_path"] == "swe_rebench_v2/tasks/a__b-1"
    assert rows[0]["metadata"]["source"] == {
        "repo": "a/b",
        "language": "python",
        "license": "MIT",
    }
    assert sorted(path.name for path in (root / "tasks").iterdir()) == [
        "a__b-1",
        "e__f-1",
    ]
    assert read_jsonl(root / "all.converted.jsonl") == rows
    skipped = capsys.readouterr().out
    assert "skipped 3 rows" in skipped
    for reason in ("unsupported language 'go'", "quality grade", "requires GPU/TPU"):
        assert f"1  {reason}" in skipped


def test_convert_stops_at_the_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_fork(monkeypatch)
    source = _source(limit=1)
    monkeypatch.setattr(
        source,
        "rows",
        lambda revision: iter(
            {"instance_id": f"r/p-{i}", "repo": "r/p", "language": "python"}
            for i in range(5)
        ),
    )
    assert len(source.convert(tmp_path, revision="deadbeef")) == 1
