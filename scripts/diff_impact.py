#!/usr/bin/env python3
"""Report which model/config classes and tutorials are affected by a diff.

Usage:
    git diff --cached | uv run scripts/diff_impact.py
    uv run scripts/diff_impact.py --diff-file /tmp/change.diff
    uv run scripts/diff_impact.py < /tmp/change.diff

The script reads a unified diff, extracts changed paths, maps those paths to
public classes in the repo, then finds tutorials whose `api_classes` include
any affected class.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
TUTORIAL_SRC_ROOT = REPO_ROOT / "tutorials" / "tutorial_generator"
TUTORIAL_OUTPUT_ROOT = REPO_ROOT / "tutorials"
MODEL_PACKAGE_ROOT = REPO_ROOT / "modal_training_gym" / "common" / "models"
SLIME_RECIPE_PACKAGE_ROOT = (
    REPO_ROOT / "modal_training_gym" / "train_recipes" / "slime_recipe"
)
MILES_RECIPE_PACKAGE_ROOT = (
    REPO_ROOT / "modal_training_gym" / "train_recipes" / "miles_recipe"
)

VALIDATION_BACKEND_ROOT = REPO_ROOT / "scripts" / "validation_backends"

# Touching the shared validation harness or shared training plumbing
# invalidates every model, so a diff that hits any of these forces a full
# re-validation.
SHARED_VALIDATION_HARNESS_PATHS = frozenset(
    {
        REPO_ROOT / "scripts" / "validate_model_configs.py",
        REPO_ROOT / "scripts" / "diff_impact.py",
        VALIDATION_BACKEND_ROOT / "__init__.py",
        REPO_ROOT / "modal_training_gym" / "common" / "models" / "validation.py",
        REPO_ROOT / "modal_training_gym" / "common" / "train.py",
        REPO_ROOT / "modal_training_gym" / "common" / "train_result.py",
    }
)

# Per-framework harness paths invalidate only the models that train on that
# framework. Without this split, adding a miles model would make every
# miles-only change re-validate the whole slime set.
FRAMEWORK_VALIDATION_HARNESS_PATHS: dict[str, frozenset[Path]] = {
    "slime": frozenset(
        {
            REPO_ROOT / "modal_training_gym" / "frameworks" / "slime" / "launcher.py",
            VALIDATION_BACKEND_ROOT / "slime.py",
        }
    ),
    "miles": frozenset(
        {
            REPO_ROOT / "modal_training_gym" / "frameworks" / "miles" / "launcher.py",
            VALIDATION_BACKEND_ROOT / "miles.py",
        }
    ),
}


@dataclass(frozen=True)
class TutorialInfo:
    slug: str
    source_path: Path
    generated_path: Path
    summary: str
    framework: str
    api_classes: tuple[str, ...]


@dataclass(frozen=True)
class ImpactReport:
    affected_classes: tuple[str, ...]
    affected_tutorials: tuple[tuple[str, str, tuple[str, ...]], ...]
    rerun_all_tutorials: bool = False


def _parse_tutorial_metadata(source: str) -> dict | None:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "TUTORIAL_METADATA":
                try:
                    value = ast.literal_eval(node.value)
                except (ValueError, SyntaxError):
                    return None
                return value if isinstance(value, dict) else None
    return None


@lru_cache(maxsize=1)
def _load_tutorial_index() -> tuple[dict[str, TutorialInfo], dict[str, set[str]]]:
    tutorials: dict[str, TutorialInfo] = {}
    class_to_tutorials: dict[str, set[str]] = defaultdict(set)

    for source_path in sorted(TUTORIAL_SRC_ROOT.rglob("*.py")):
        if source_path.name == "__init__.py":
            continue
        metadata = _parse_tutorial_metadata(source_path.read_text())
        if metadata is None:
            continue
        rel = source_path.relative_to(TUTORIAL_SRC_ROOT).with_suffix("")
        slug = "/".join(rel.parts)
        generated_path = TUTORIAL_OUTPUT_ROOT / rel / f"{rel.name}.py"
        info = TutorialInfo(
            slug=slug,
            source_path=source_path,
            generated_path=generated_path,
            summary=str(metadata.get("summary", rel.name)),
            framework=str(metadata.get("framework", "")),
            api_classes=tuple(str(name) for name in metadata.get("api_classes", [])),
        )
        tutorials[slug] = info
        for class_name in info.api_classes:
            class_to_tutorials[class_name].add(slug)

    return tutorials, class_to_tutorials


def _parse_public_definitions(path: Path) -> set[str]:
    try:
        tree = ast.parse(path.read_text())
    except (OSError, SyntaxError):
        return set()

    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if not node.name.startswith("_"):
                names.add(node.name)
    return names


@lru_cache(maxsize=None)
def _package_public_definitions(package_root: str) -> set[str]:
    root = Path(package_root)
    names: set[str] = set()
    for path in root.rglob("*.py"):
        if path.name == "__init__.py":
            continue
        names.update(_parse_public_definitions(path))
    return names


@lru_cache(maxsize=1)
def _model_index() -> tuple[dict[str, frozenset[str]], dict[str, frozenset[str]]]:
    """Map defining classes to the model names they gate, and frameworks to theirs.

    A model is validated by ``validate_model_configs.py check --model <name>``,
    where ``<name>`` is the registry's short name (e.g. ``Qwen3-0.6B``). Both
    the model's ``ModelConfig`` subclass and the recipe class its framework
    returns from ``get_base_recipe`` gate that model, so a change to either
    re-validates it.

    Only PR-gating entries are indexed. A model registered with
    ``ci_enabled=False`` (e.g. Kimi on 16 x 8 H200) must never reach a PR
    matrix, so no diff can select it.
    """
    from modal_training_gym.common.models.validation import (
        Framework,
        _ValidationConfig,
    )
    from modal_training_gym.train_recipes.miles_recipe import MilesRecipe
    from modal_training_gym.train_recipes.slime_recipe import SlimeRecipe

    recipe_classes = {
        Framework.SLIME: SlimeRecipe,
        Framework.MILES: MilesRecipe,
    }

    class_to_models: dict[str, set[str]] = defaultdict(set)
    framework_to_models: dict[str, set[str]] = defaultdict(set)
    for config in _ValidationConfig.select():
        class_to_models[config.model_config.__name__].add(config.name)
        framework_to_models[config.framework.value].add(config.name)
        recipe = recipe_classes[config.framework].get_base_recipe(config.model_config())
        if recipe is not None:
            class_to_models[type(recipe).__name__].add(config.name)

    return (
        {name: frozenset(models) for name, models in class_to_models.items()},
        {name: frozenset(models) for name, models in framework_to_models.items()},
    )


def affected_models(diff_text: str) -> tuple[str, ...]:
    """Model names (validate ``--model`` args) impacted by a diff.

    Importing ``modal_training_gym`` is deferred to here so the tutorial-only
    paths through ``analyze_diff`` stay import-free.
    """
    changed_paths = _paths_from_diff(diff_text)
    class_to_models, framework_to_models = _model_index()
    all_models = {model for models in framework_to_models.values() for model in models}

    if any(path in SHARED_VALIDATION_HARNESS_PATHS for path in changed_paths):
        return tuple(sorted(all_models))

    models: set[str] = set()
    for framework, harness_paths in FRAMEWORK_VALIDATION_HARNESS_PATHS.items():
        if any(path in harness_paths for path in changed_paths):
            models.update(framework_to_models.get(framework, frozenset()))

    report = analyze_diff(diff_text)
    for class_name in report.affected_classes:
        models.update(class_to_models.get(class_name, frozenset()))
    return tuple(sorted(models))


def _generated_tutorial_source(path: Path) -> Path | None:
    try:
        rel = path.relative_to(TUTORIAL_OUTPUT_ROOT)
    except ValueError:
        return None

    if len(rel.parts) < 2:
        return None
    stem = rel.stem
    bucket = rel.parts[0]
    source = TUTORIAL_SRC_ROOT / bucket / f"{stem}.py"
    return source if source.exists() else None


def _paths_from_diff(diff_text: str) -> set[Path]:
    paths: set[Path] = set()
    for line in diff_text.splitlines():
        if line.startswith("diff --git "):
            match = re.match(r"diff --git a/(.+?) b/(.+)$", line)
            if not match:
                continue
            for raw in match.groups():
                if raw != "/dev/null":
                    paths.add(REPO_ROOT / raw)
            continue

        if line.startswith("rename from ") or line.startswith("rename to "):
            raw = line.split(" ", 2)[2].strip()
            if raw != "/dev/null":
                paths.add(REPO_ROOT / raw)
            continue

        if line.startswith("+++ ") or line.startswith("--- "):
            raw = line[4:].strip()
            if raw.startswith("a/") or raw.startswith("b/"):
                raw = raw[2:]
            if raw != "/dev/null":
                paths.add(REPO_ROOT / raw)

    return paths


def _tutorial_source_from_generated(path: Path) -> Path | None:
    source = _generated_tutorial_source(path)
    if source is not None:
        return source
    return None


def _classes_for_path(path: Path) -> set[str]:
    if path == REPO_ROOT / "tutorials" / "generate_tutorial.py":
        return set()

    if path == TUTORIAL_SRC_ROOT / "__init__.py":
        return set()

    if path.is_relative_to(TUTORIAL_SRC_ROOT):
        info = _load_tutorial_index()[0].get(
            "/".join(path.relative_to(TUTORIAL_SRC_ROOT).with_suffix("").parts)
        )
        return set(info.api_classes) if info else set()

    generated_source = _tutorial_source_from_generated(path)
    if generated_source is not None:
        tutorial_slug = "/".join(
            generated_source.relative_to(TUTORIAL_SRC_ROOT).with_suffix("").parts
        )
        info = _load_tutorial_index()[0].get(tutorial_slug)
        return set(info.api_classes) if info else set()

    if path.is_relative_to(MODEL_PACKAGE_ROOT):
        if path.name in {"base.py", "__init__.py"}:
            return set(_package_public_definitions(str(MODEL_PACKAGE_ROOT)))
        return _parse_public_definitions(path)

    if path.is_relative_to(SLIME_RECIPE_PACKAGE_ROOT):
        if path.name in {"recipe.py", "__init__.py"}:
            return set(_package_public_definitions(str(SLIME_RECIPE_PACKAGE_ROOT)))
        return _parse_public_definitions(path)

    if path.is_relative_to(MILES_RECIPE_PACKAGE_ROOT):
        if path.name in {"recipe.py", "__init__.py"}:
            return set(_package_public_definitions(str(MILES_RECIPE_PACKAGE_ROOT)))
        return _parse_public_definitions(path)

    if path.suffix == ".py" and path.exists():
        return _parse_public_definitions(path)

    return set()


def analyze_diff(diff_text: str) -> ImpactReport:
    tutorials, class_to_tutorials = _load_tutorial_index()
    changed_paths = _paths_from_diff(diff_text)

    affected_classes: set[str] = set()
    affected_tutorial_reasons: dict[str, set[str]] = defaultdict(set)
    rerun_all_tutorials = False

    for path in changed_paths:
        if path == REPO_ROOT / "tutorials" / "generate_tutorial.py" or (
            path == TUTORIAL_SRC_ROOT / "__init__.py"
        ):
            rerun_all_tutorials = True
            continue

        if path.is_relative_to(TUTORIAL_SRC_ROOT):
            slug = "/".join(path.relative_to(TUTORIAL_SRC_ROOT).with_suffix("").parts)
            info = tutorials.get(slug)
            if info is not None:
                affected_tutorial_reasons[slug].add("tutorial source changed")
            continue

        generated_source = _generated_tutorial_source(path)
        if generated_source is not None:
            slug = "/".join(
                generated_source.relative_to(TUTORIAL_SRC_ROOT).with_suffix("").parts
            )
            info = tutorials.get(slug)
            if info is not None:
                affected_tutorial_reasons[slug].add("generated tutorial changed")
            continue

        classes = _classes_for_path(path)
        affected_classes.update(classes)
        for class_name in classes:
            for slug in class_to_tutorials.get(class_name, set()):
                affected_tutorial_reasons[slug].add(class_name)

    if rerun_all_tutorials:
        for slug, info in tutorials.items():
            affected_tutorial_reasons[slug].add("tutorial generator changed")
        affected_classes.update(
            cls for info in tutorials.values() for cls in info.api_classes
        )

    affected_tutorials = tuple(
        sorted(
            (
                slug,
                tutorials[slug].summary,
                tuple(sorted(reasons)),
            )
            for slug, reasons in affected_tutorial_reasons.items()
            if slug in tutorials
        )
    )

    return ImpactReport(
        affected_classes=tuple(sorted(affected_classes)),
        affected_tutorials=affected_tutorials,
        rerun_all_tutorials=rerun_all_tutorials,
    )


def _format_report(report: ImpactReport) -> str:
    lines: list[str] = []
    lines.append("Affected classes:")
    if report.affected_classes:
        for name in report.affected_classes:
            lines.append(f"- {name}")
    else:
        lines.append("- none")

    lines.append("")
    lines.append("Tutorials to rerun:")
    if report.affected_tutorials:
        for slug, summary, reasons in report.affected_tutorials:
            reason_text = ", ".join(reasons)
            if reason_text:
                lines.append(f"- {slug} — {summary} ({reason_text})")
            else:
                lines.append(f"- {slug} — {summary}")
    else:
        lines.append("- none")

    return "\n".join(lines)


def _report_to_json(report: ImpactReport) -> str:
    payload = {
        "affected_classes": list(report.affected_classes),
        "rerun_all_tutorials": report.rerun_all_tutorials,
        "affected_tutorials": [
            {
                "slug": slug,
                "summary": summary,
                "reasons": list(reasons),
            }
            for slug, summary, reasons in report.affected_tutorials
        ],
    }
    return json.dumps(payload, indent=2, sort_keys=True)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Analyze a diff and report affected model classes and tutorials."
    )
    parser.add_argument(
        "--diff-file",
        type=Path,
        help="Read the diff from a file instead of stdin.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of a human-readable summary.",
    )
    parser.add_argument(
        "--models",
        action="store_true",
        help="Emit a JSON array of model names affected by the diff, "
        "compatible with scripts/validate_model_configs.py check --model.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.diff_file is not None:
        diff_text = args.diff_file.read_text()
    else:
        diff_text = sys.stdin.read()

    if args.models:
        print(json.dumps(list(affected_models(diff_text))))
        return 0

    report = analyze_diff(diff_text)
    output = _report_to_json(report) if args.json else _format_report(report)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
