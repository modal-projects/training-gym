"""Python-native Harbor task builder.

Replaces hand-written TOML / shell strings with a dataclass that
serializes a task directory in the format Harbor expects::

    task_dir/
    ├── instruction.md
    ├── task.toml
    ├── environment/
    │   └── Dockerfile
    ├── tests/
    │   ├── verify.py   (user-provided)
    │   └── test.sh     (auto-generated wrapper)
    └── solution/       (optional)
        └── solve.sh

Usage::

    task = HarborTask(
        name="mbpp_0601",
        instruction="Write a function ...",
        tags=["mbpp", "python"],
        metadata={"author": "MBPP", "task_id": 601},
    )
    task.write(tasks_dir)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


_DEFAULT_DOCKERFILE = "FROM python:3.11-slim\nWORKDIR /workspace\n"

_TEST_SH_TEMPLATE = """\
#!/usr/bin/env bash
set -euo pipefail
mkdir -p /logs/verifier
python3 /workspace/../tests/verify.py 2>&1 || true
if [ ! -f /logs/verifier/reward.json ]; then
    echo '{"reward": 0.0, "pass_rate": 0.0, "error": "verifier crashed"}' > /logs/verifier/reward.json
fi
"""


@dataclass
class HarborTask:
    """A single Harbor task directory."""

    name: str
    instruction: str

    # task.toml fields
    version: str = "0.1"
    difficulty: str = "medium"
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    agent_timeout: int = 180
    verifier_timeout: int = 180
    cpus: int = 1
    memory_mb: int = 2048
    storage_mb: int = 2048
    allow_internet: bool = False

    # environment
    dockerfile: str = _DEFAULT_DOCKERFILE

    # verification (verify.py content)
    verify_script: str = ""

    # optional reference solution (solve.sh content)
    solution_script: str = ""

    def _to_toml(self) -> str:
        lines: list[str] = []

        lines.append("[task]")
        lines.append(f'version = "{self.version}"')
        lines.append(f'difficulty = "{self.difficulty}"')
        lines.append("")

        if self.metadata or self.tags:
            lines.append("[task.metadata]")
            for k, v in self.metadata.items():
                lines.append(f"{k} = {_toml_value(v)}")
            if self.tags:
                tag_strs = ", ".join(f'"{t}"' for t in self.tags)
                lines.append(f"tags = [{tag_strs}]")
            lines.append("")

        lines.append("[timeouts]")
        lines.append(f"agent = {self.agent_timeout}")
        lines.append(f"verifier = {self.verifier_timeout}")
        lines.append("")

        lines.append("[environment]")
        lines.append(f"cpus = {self.cpus}")
        lines.append(f"memory_mb = {self.memory_mb}")
        lines.append(f"storage_mb = {self.storage_mb}")
        lines.append(f"allow_internet = {'true' if self.allow_internet else 'false'}")
        lines.append("")

        return "\n".join(lines)

    def write(self, tasks_dir: Path) -> Path:
        """Write this task to ``tasks_dir / self.name``."""
        task_dir = tasks_dir / self.name
        task_dir.mkdir(parents=True, exist_ok=True)

        (task_dir / "instruction.md").write_text(self.instruction)
        (task_dir / "task.toml").write_text(self._to_toml())

        env_dir = task_dir / "environment"
        env_dir.mkdir(exist_ok=True)
        (env_dir / "Dockerfile").write_text(self.dockerfile)

        if self.verify_script:
            tests_dir = task_dir / "tests"
            tests_dir.mkdir(exist_ok=True)
            (tests_dir / "verify.py").write_text(self.verify_script)
            (tests_dir / "test.sh").write_text(_TEST_SH_TEMPLATE)

        if self.solution_script:
            solution_dir = task_dir / "solution"
            solution_dir.mkdir(exist_ok=True)
            (solution_dir / "solve.sh").write_text(self.solution_script)

        return task_dir


def _toml_value(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return str(v)
    if isinstance(v, str):
        return json.dumps(v)
    if isinstance(v, list):
        items = ", ".join(_toml_value(x) for x in v)
        return f"[{items}]"
    return json.dumps(str(v))
