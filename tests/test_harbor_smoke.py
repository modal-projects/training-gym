"""Smoke test for the Harbor framework.

Validates:
- HarborFrameworkConfig construction and CLI arg generation
- HarborConfig app building
- JSONL builder
- Deploying prepare_dataset on Modal (creates tasks + JSONL on a volume)

Run locally (no Modal):
    uv run python tests/test_harbor_smoke.py

Run with Modal deploy (prepare_dataset):
    uv run modal run tests/test_harbor_smoke.py
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from modal_training_gym.common.dataset import DatasetConfig
from modal_training_gym.common.models import Qwen3_4B
from modal_training_gym.frameworks.harbor import (
    HarborConfig,
    HarborFrameworkConfig,
    HarborTask,
)
from modal_training_gym.frameworks.harbor.launcher import _build_training_jsonl
from modal_training_gym.frameworks.miles import MilesFrameworkConfig


def test_config_inheritance():
    cfg = HarborFrameworkConfig()
    assert isinstance(cfg, MilesFrameworkConfig), (
        "HarborFrameworkConfig must extend MilesFrameworkConfig"
    )
    assert cfg.gpu == "H100"
    assert cfg.agent_import_path == ""
    assert cfg.sandbox_timeout_secs == 1800
    print("  PASS  test_config_inheritance")


def test_cli_args_exclude_harbor_fields():
    cfg = HarborFrameworkConfig(
        agent_import_path="test:Agent",
        agent_model_name="test-model",
        sandbox_timeout_secs=600,
        task_root="/data/my_tasks",
        environment_import_path="env:MyEnv",
        harbor_install_command="pip install harbor",
    )
    args = cfg.cli_args()
    for arg in args:
        for forbidden in (
            "agent",
            "sandbox",
            "harbor",
            "environment-import",
            "task-root",
            "task-glob",
            "instruction-path",
        ):
            assert forbidden not in arg, f"Harbor field leaked to CLI: {arg}"
    print("  PASS  test_cli_args_exclude_harbor_fields")


def test_harbor_config_construction():
    cfg = HarborFrameworkConfig(
        agent_import_path="my_agent:MyAgent",
    )
    harbor = HarborConfig(
        model=Qwen3_4B(),
        framework_config=cfg,
    )
    assert harbor.framework_config.agent_import_path == "my_agent:MyAgent"
    assert harbor.model is not None
    assert harbor.model.model_name == "Qwen/Qwen3-4B"
    assert harbor.framework_config.input_key == "prompt"
    assert harbor.framework_config.num_rollout == 200
    print("  PASS  test_harbor_config_construction")


def test_build_app():
    cfg = HarborFrameworkConfig(
        agent_import_path="my_agent:MyAgent",
    )
    harbor = HarborConfig(
        model=Qwen3_4B(),
        dataset=DatasetConfig(),
        framework_config=cfg,
    )
    app = harbor.build_app(name="test-harbor-smoke")
    expected = {"download_model", "prepare_dataset", "train_multi_node", "eval_harbor"}
    actual = set(app.registered_functions.keys())
    missing = expected - actual
    assert not missing, f"Missing functions: {missing}"
    print("  PASS  test_build_app")


def test_jsonl_builder():
    with tempfile.TemporaryDirectory() as tmpdir:
        tasks_dir = Path(tmpdir) / "tasks"
        tasks_dir.mkdir()

        for i in range(3):
            task = tasks_dir / f"task_{i:03d}"
            task.mkdir()
            (task / "instruction.md").write_text(f"Solve problem {i}.")
            (task / "tests").mkdir()
            (task / "tests" / "test.sh").write_text("echo OK")

        # Task without instruction (should be skipped)
        (tasks_dir / "task_empty").mkdir()

        output = Path(tmpdir) / "train.jsonl"
        count = _build_training_jsonl(
            str(tasks_dir),
            output,
            agent_kwargs={"temperature": 0.7},
        )

        assert count == 3, f"Expected 3 tasks, got {count}"
        lines = output.read_text().strip().split("\n")
        assert len(lines) == 3

        for i, line in enumerate(lines):
            record = json.loads(line)
            assert "prompt" in record
            assert "metadata" in record
            assert "harbor_task_path" in record["metadata"]
            assert "harbor_task_name" in record["metadata"]
            assert record["metadata"]["harbor_agent_kwargs"]["temperature"] == 0.7

    print("  PASS  test_jsonl_builder")


def run_local_tests():
    test_config_inheritance()
    test_cli_args_exclude_harbor_fields()
    test_harbor_config_construction()
    test_build_app()
    test_jsonl_builder()
    print("\nAll 5 test(s) passed.")


# ── Modal deploy test ────────────────────────────────────────────────────────

DATA_PATH = Path("/data")


class HelloWorldDataset(DatasetConfig):
    """Creates minimal Harbor tasks for smoke testing."""

    def prepare(self):
        tasks_dir = DATA_PATH / "tasks"
        tasks_dir.mkdir(parents=True, exist_ok=True)

        HarborTask(
            name="hello-world",
            instruction="Create a file called hello.txt with 'Hello, world!' as the content.",
            difficulty="easy",
            agent_timeout=120,
            verifier_timeout=120,
            dockerfile="FROM python:3.12-slim\nWORKDIR /app\n",
        ).write(tasks_dir)

        print("Prepared hello-world task")


harbor = HarborConfig(
    model=Qwen3_4B(),
    dataset=HelloWorldDataset(),
    framework_config=HarborFrameworkConfig(
        agent_import_path="placeholder:Agent",
    ),
)

app = harbor.build_app(name="harbor-smoke-test")


if __name__ == "__main__":
    run_local_tests()
