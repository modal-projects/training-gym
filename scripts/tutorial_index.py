from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TUTORIALS_DIR = REPO_ROOT / "tutorials"
FIELD_PATTERN = re.compile(r"^# ([a-z_]+):\s*(.*)$")
DEP_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")
ORDER_PATTERN = re.compile(r"^\d+$")
MAX_SAFE_INTEGER = (1 << 53) - 1


@dataclass(frozen=True)
class TutorialEntry:
    path: Path
    slug: str
    order: int
    title: str
    deps: tuple[str, ...]

    @property
    def run_command(self) -> str:
        with_args = " ".join(f"--with {dependency}" for dependency in self.deps)
        prefix = f"uv run {with_args}" if with_args else "uv run"
        return f"{prefix} tutorials/{self.slug}.py"


def parse_tutorial(path: Path) -> TutorialEntry:
    lines = path.read_text().splitlines()
    if not lines or lines[0] != "# ---":
        raise ValueError(f"{path} must start with tutorial frontmatter")

    try:
        frontmatter_end = lines.index("# ---", 1)
    except ValueError as exc:
        raise ValueError(f"{path} has unterminated tutorial frontmatter") from exc

    fields: dict[str, str] = {}
    for line in lines[1:frontmatter_end]:
        match = FIELD_PATTERN.fullmatch(line)
        if match is None:
            raise ValueError(f"{path} has invalid frontmatter line: {line!r}")
        name, value = match.groups()
        if name not in {"order", "deps"}:
            raise ValueError(f"{path} has unsupported frontmatter field: {name}")
        if name in fields:
            raise ValueError(f"{path} has duplicate frontmatter field: {name}")
        fields[name] = value

    order_text = fields.get("order")
    if order_text is None or ORDER_PATTERN.fullmatch(order_text) is None:
        raise ValueError(f"{path} frontmatter requires a non-negative integer order")
    order = int(order_text)
    if order > MAX_SAFE_INTEGER:
        raise ValueError(f"{path} frontmatter order exceeds the safe integer range")

    deps = tuple(
        dependency.strip()
        for dependency in fields.get("deps", "").split(",")
        if dependency.strip()
    )
    if len(deps) != len(set(deps)):
        raise ValueError(f"{path} frontmatter deps must be unique")
    invalid_deps = [
        dependency for dependency in deps if not DEP_PATTERN.fullmatch(dependency)
    ]
    if invalid_deps:
        raise ValueError(f"{path} has invalid frontmatter deps: {invalid_deps}")

    title_line = next(
        (line for line in lines[frontmatter_end + 1 :] if line.startswith("# # ")),
        None,
    )
    if title_line is None:
        raise ValueError(f"{path} is missing an H1 heading")

    return TutorialEntry(
        path=path,
        slug=path.stem,
        order=order,
        title=title_line.removeprefix("# # ").strip(),
        deps=deps,
    )


def load_tutorial_index(
    tutorials_dir: Path = TUTORIALS_DIR,
) -> tuple[TutorialEntry, ...]:
    entries = tuple(
        sorted(
            (parse_tutorial(path) for path in tutorials_dir.glob("*.py")),
            key=lambda entry: (entry.order, entry.slug),
        )
    )
    orders = [entry.order for entry in entries]
    if orders != list(range(len(entries))):
        raise ValueError(f"Tutorial orders must be contiguous from 0: {orders}")
    return entries
