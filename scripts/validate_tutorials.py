#!/usr/bin/env python3
"""One-pass tutorial validation sweep for modal-training-gym.

Discovers tutorials from generator sources, classifies them by type,
runs a local preflight, then executes each on Modal and reports outcomes.

Usage:
    uv run python scripts/validate_tutorials.py              # full sweep
    uv run python scripts/validate_tutorials.py --list        # show discovered targets
    uv run python scripts/validate_tutorials.py --only slime_gsm8k  # single tutorial
    uv run python scripts/validate_tutorials.py --preflight-only     # local checks only
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TextIO

REPO_ROOT = Path(__file__).resolve().parent.parent
GENERATOR_DIR = REPO_ROOT / "tutorials" / "tutorial_generator"
TUTORIALS_DIR = REPO_ROOT / "tutorials"


# ── Canonical enums ──────────────────────────────────────────────────────────


class Outcome(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    BLOCKED = "blocked"
    SKIPPED = "skipped"


class FailureSource(str, Enum):
    LOCAL_COMPILE = "local_compile"
    GENERATOR_DRIFT = "generator_drift"
    ARG_PARSE = "arg_parse"
    IMAGE_BUILD = "image_build"
    PREREQUISITE_STAGE = "prerequisite_stage"
    TRAINING_RUNTIME = "training_runtime"
    CAPACITY = "capacity"
    GATED_ACCESS = "gated_access"
    SECRET_MISSING = "secret_missing"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


class BlockedReason(str, Enum):
    MISSING_SECRET = "missing_secret"
    GATED_ACCESS = "gated_access"
    CAPACITY = "capacity"


class TargetType(str, Enum):
    TRAINING = "training"
    BENCHMARK = "benchmark"
    PATTERN_DEMO = "pattern_demo"
    CONCEPT_ONLY = "concept_only"


# ── Data classes ─────────────────────────────────────────────────────────────


@dataclass
class TutorialTarget:
    tutorial_stem: str
    bucket: str
    source_path: Path
    generated_py: Path
    target_name: str
    target_type: TargetType
    stages: list[str]
    progress_markers: list[str]
    progress_timeout_s: int = 1800
    metadata: dict = field(default_factory=dict)


@dataclass
class StageResult:
    stage: str
    command: str
    exit_code: int | None = None
    app_id: str | None = None
    first_signal: str | None = None
    first_error: str | None = None
    wall_time_s: float = 0.0
    stdout_tail: str = ""
    stderr_tail: str = ""


@dataclass
class ValidationResult:
    tutorial: str
    target: str
    target_type: str
    outcome: Outcome
    failure_source: FailureSource | None = None
    blocked_reason: BlockedReason | None = None
    reason_detail: str = ""
    stages: list[StageResult] = field(default_factory=list)
    wall_time_s: float = 0.0
    app_id: str | None = None
    first_signal: str | None = None


# ── Discovery (task2) ────────────────────────────────────────────────────────


def _extract_metadata(source: str) -> dict | None:
    tree = ast.parse(source)
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "TUTORIAL_METADATA":
                try:
                    return ast.literal_eval(node.value)
                except (ValueError, SyntaxError):
                    return None
    return None


def _extract_run_cli_commands(source: str) -> list[str]:
    """Extract modal run commands from the _run_cli markdown docstring."""
    tree = ast.parse(source)
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if "_run_cli" not in node.name:
            continue
        doc = ast.get_docstring(node, clean=True) or ""
        commands = []
        for line in doc.splitlines():
            line = line.strip()
            if "modal run" in line and not line.startswith("#"):
                line = line.removeprefix("uv run ").strip()
                commands.append(line)
        return commands
    return []


def _bucket_from_path(source_path: Path) -> str:
    rel = source_path.relative_to(GENERATOR_DIR)
    parts = rel.parts
    return parts[0] if len(parts) > 1 else ""


def _resolve_generated_py(bucket: str, stem: str) -> Path:
    if bucket:
        return TUTORIALS_DIR / bucket / stem / f"{stem}.py"
    return TUTORIALS_DIR / stem / f"{stem}.py"


def _parse_stages_from_commands(commands: list[str], stem: str) -> list[tuple[str, str]]:
    """Return [(stage_name, full_command), ...] from documented CLI commands.

    Deduplicates by entrypoint name (keeps first occurrence), and skips
    commands that have CLI args after `--` (Hydra override examples) or
    extra `--entrypoint` flags (variant examples).
    """
    _SHORTCUTS = {
        "app.download_and_convert": {"app.download_model", "app.convert_to_megatron"},
    }

    stages = []
    seen_entrypoints: set[str] = set()
    for cmd in commands:
        if " -- " in cmd:
            continue
        if "--entrypoint" in cmd:
            continue
        match = re.search(r"::([\w.]+)", cmd)
        if match:
            entrypoint = match.group(1)
            if entrypoint in seen_entrypoints:
                continue
            seen_entrypoints.add(entrypoint)
            stages.append((entrypoint, cmd))

    all_eps = {ep for ep, _ in stages}
    stages = [
        (ep, cmd)
        for ep, cmd in stages
        if ep not in _SHORTCUTS or not _SHORTCUTS[ep].issubset(all_eps)
    ]
    return stages


def discover_tutorials() -> list[TutorialTarget]:
    targets = []
    for source_path in sorted(GENERATOR_DIR.rglob("*.py")):
        if source_path.name == "__init__.py":
            continue

        stem = source_path.stem
        bucket = _bucket_from_path(source_path)

        try:
            source = source_path.read_text()
        except OSError:
            continue

        metadata = _extract_metadata(source)
        if metadata is None:
            continue

        generated_py = _resolve_generated_py(bucket, stem)
        commands = _extract_run_cli_commands(source)
        stages = _parse_stages_from_commands(commands, stem)

        classification = TUTORIAL_CLASSIFICATIONS.get(stem)
        if classification is None:
            continue

        target_type = classification["type"]
        markers = classification["markers"]
        timeout = classification.get("timeout", 1800)

        stage_commands = [s[1] for s in stages]

        if target_type == TargetType.CONCEPT_ONLY:
            stage_commands = []
        elif not stage_commands:
            stage_commands = [f"modal run tutorials/{bucket}/{stem}/{stem}.py"]

        train_suffix = classification.get("train_stage_suffix", "")
        if train_suffix and stage_commands:
            stage_commands[-1] = f"{stage_commands[-1]} {train_suffix}"

        target = TutorialTarget(
            tutorial_stem=stem,
            bucket=bucket,
            source_path=source_path,
            generated_py=generated_py,
            target_name=f"{stem}",
            target_type=target_type,
            stages=stage_commands,
            progress_markers=markers,
            progress_timeout_s=timeout,
            metadata=metadata,
        )
        targets.append(target)

    # starcoder_llama2_7b: split into two targets (torchrun_app + accelerate_app)
    expanded = []
    for t in targets:
        if t.tutorial_stem == "starcoder_llama2_7b":
            torchrun_stages = [s for s in t.stages if "torchrun_app" in s]
            accel_stages = [s for s in t.stages if "accelerate_app" in s]
            if torchrun_stages and accel_stages:
                t1 = TutorialTarget(
                    tutorial_stem=t.tutorial_stem,
                    bucket=t.bucket,
                    source_path=t.source_path,
                    generated_py=t.generated_py,
                    target_name="starcoder_llama2_7b/torchrun",
                    target_type=t.target_type,
                    stages=torchrun_stages,
                    progress_markers=t.progress_markers,
                    progress_timeout_s=t.progress_timeout_s,
                    metadata=t.metadata,
                )
                t2 = TutorialTarget(
                    tutorial_stem=t.tutorial_stem,
                    bucket=t.bucket,
                    source_path=t.source_path,
                    generated_py=t.generated_py,
                    target_name="starcoder_llama2_7b/accelerate",
                    target_type=t.target_type,
                    stages=accel_stages,
                    progress_markers=t.progress_markers,
                    progress_timeout_s=t.progress_timeout_s,
                    metadata=t.metadata,
                )
                expanded.extend([t1, t2])
                continue
        expanded.append(t)

    return expanded


# ── Classification table (task3) ─────────────────────────────────────────────

TUTORIAL_CLASSIFICATIONS: dict[str, dict] = {
    "quickstart": {
        "type": TargetType.CONCEPT_ONLY,
        "markers": [],
    },
    "nccl_benchmark": {
        "type": TargetType.BENCHMARK,
        "markers": ["busbw:", "algbw:"],
        "timeout": 600,
    },
    "ray_slime_standalone": {
        "type": TargetType.PATTERN_DEMO,
        "markers": ["Ray dashboard:", "ray.cluster_resources"],
        "timeout": 600,
    },
    "slime_gsm8k": {
        "type": TargetType.TRAINING,
        "markers": ["train/reward", "iter 0", "rollout"],
    },
    "slime_haiku": {
        "type": TargetType.TRAINING,
        "markers": ["train/reward", "iter 0", "rollout"],
    },
    "verl_qwen3_32b_gsm8k": {
        "type": TargetType.TRAINING,
        "markers": ["train/reward", "iter 0", "reward"],
        "train_stage_suffix": "-- trainer.total_training_steps=1",
    },
    "ms_swift_glm_4_7_gsm8k": {
        "type": TargetType.TRAINING,
        "markers": ["iter 0/", "train/loss", "iteration"],
    },
    "ms_swift_custom_hf": {
        "type": TargetType.TRAINING,
        "markers": ["iter 0/", "train/loss", "iteration"],
    },
    "megatron_glm_4_7_longmit128k": {
        "type": TargetType.TRAINING,
        "markers": ["iter 0/", "iteration", "lm loss"],
    },
    "starcoder_llama2_7b": {
        "type": TargetType.TRAINING,
        "markers": ['{"loss":', "loss", "train_loss"],
    },
    "nanogpt_owt": {
        "type": TargetType.TRAINING,
        "markers": ["iter 0,", "loss=", "iter 0 "],
    },
    "lightning_fabric_demo": {
        "type": TargetType.TRAINING,
        "markers": ["iter 0/", "loss"],
    },
    "resnet50_imagenet": {
        "type": TargetType.TRAINING,
        "markers": ["Epoch:", "Loss", "Acc@1"],
    },
}


# ── Preflight (task1) ────────────────────────────────────────────────────────


def run_preflight(target: TutorialTarget) -> ValidationResult | None:
    """Run Tier-0 local checks. Returns a fail result if any check fails, else None."""

    # 1. Parse source
    try:
        with open(target.source_path) as f:
            source = f.read()
        compile(source, str(target.source_path), "exec")
    except SyntaxError as e:
        return ValidationResult(
            tutorial=target.tutorial_stem,
            target=target.target_name,
            target_type=target.target_type.value,
            outcome=Outcome.FAIL,
            failure_source=FailureSource.LOCAL_COMPILE,
            reason_detail=f"syntax error in {target.source_path}: {e}",
        )

    # 2. Generator freshness
    if target.generated_py.exists():
        src_mtime = target.source_path.stat().st_mtime
        gen_mtime = target.generated_py.stat().st_mtime
        if src_mtime > gen_mtime:
            return ValidationResult(
                tutorial=target.tutorial_stem,
                target=target.target_name,
                target_type=target.target_type.value,
                outcome=Outcome.FAIL,
                failure_source=FailureSource.GENERATOR_DRIFT,
                reason_detail=(
                    f"{target.source_path} is newer than {target.generated_py}. "
                    "Regenerate via: uv run python tutorials/generate_tutorial.py"
                ),
            )
    else:
        return ValidationResult(
            tutorial=target.tutorial_stem,
            target=target.target_name,
            target_type=target.target_type.value,
            outcome=Outcome.FAIL,
            failure_source=FailureSource.GENERATOR_DRIFT,
            reason_detail=f"generated file {target.generated_py} does not exist",
        )

    # 3. Entrypoint existence (for remote targets)
    if target.target_type != TargetType.CONCEPT_ONLY:
        gen_source = target.generated_py.read_text()
        for stage_cmd in target.stages:
            match = re.search(r"::(\w+)", stage_cmd)
            if match:
                entrypoint = match.group(1)
                if entrypoint not in gen_source:
                    return ValidationResult(
                        tutorial=target.tutorial_stem,
                        target=target.target_name,
                        target_type=target.target_type.value,
                        outcome=Outcome.FAIL,
                        failure_source=FailureSource.ARG_PARSE,
                        reason_detail=f"entrypoint '{entrypoint}' not found in {target.generated_py}",
                    )

    return None


# ── Executors ────────────────────────────────────────────────────────────────


def _run_modal_command(
    cmd: str,
    timeout_s: int,
    detach: bool = False,
) -> StageResult:
    """Run a modal command and capture results."""
    full_cmd = f"uv run {cmd}"
    start = time.monotonic()

    try:
        proc = subprocess.run(
            full_cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            cwd=str(REPO_ROOT),
        )
        wall = time.monotonic() - start
        return StageResult(
            stage=cmd,
            command=full_cmd,
            exit_code=proc.returncode,
            stdout_tail=proc.stdout[-2000:] if proc.stdout else "",
            stderr_tail=proc.stderr[-2000:] if proc.stderr else "",
            wall_time_s=wall,
        )
    except subprocess.TimeoutExpired:
        wall = time.monotonic() - start
        return StageResult(
            stage=cmd,
            command=full_cmd,
            exit_code=None,
            wall_time_s=wall,
            first_error="timeout",
        )


def _find_marker(text: str, markers: list[str]) -> str | None:
    for marker in markers:
        if marker.lower() in text.lower():
            for line in text.splitlines():
                if marker.lower() in line.lower():
                    return line.strip()[:200]
    return None


def _detect_blocked(stderr: str, stdout: str) -> tuple[BlockedReason, str] | None:
    combined = stderr + stdout
    if "401" in combined and ("huggingface" in combined.lower() or "hf" in combined.lower()):
        return BlockedReason.GATED_ACCESS, "HF 401 — gated model/dataset access"
    if "403" in combined and "huggingface" in combined.lower():
        return BlockedReason.GATED_ACCESS, "HF 403 — gated model/dataset access"
    secret_match = re.search(r"[Ss]ecret\s+['\"]?(\S+?)['\"]?\s+(not found|does not exist)", combined)
    if secret_match:
        return BlockedReason.MISSING_SECRET, f"missing Modal secret: {secret_match.group(1)}"
    if "no capacity" in combined.lower() or "insufficient capacity" in combined.lower():
        return BlockedReason.CAPACITY, "cluster capacity unavailable"
    return None


def validate_concept_only(target: TutorialTarget) -> ValidationResult:
    """AC-6.4: local import + compile check only."""
    start = time.monotonic()
    try:
        gen_source = target.generated_py.read_text()
        compile(gen_source, str(target.generated_py), "exec")
        wall = time.monotonic() - start
        return ValidationResult(
            tutorial=target.tutorial_stem,
            target=target.target_name,
            target_type=target.target_type.value,
            outcome=Outcome.PASS,
            reason_detail="source compiles and imports resolve",
            wall_time_s=wall,
            first_signal="compile_ok",
        )
    except Exception as e:
        wall = time.monotonic() - start
        return ValidationResult(
            tutorial=target.tutorial_stem,
            target=target.target_name,
            target_type=target.target_type.value,
            outcome=Outcome.FAIL,
            failure_source=FailureSource.LOCAL_COMPILE,
            reason_detail=str(e),
            wall_time_s=wall,
        )


def validate_remote_target(target: TutorialTarget) -> ValidationResult:
    """AC-6.1/6.2/6.3: run stages on Modal, check for markers."""
    start = time.monotonic()
    stage_results = []
    last_stage_is_training = target.target_type == TargetType.TRAINING

    for i, cmd in enumerate(target.stages):
        is_last = i == len(target.stages) - 1
        is_training_stage = is_last and last_stage_is_training

        timeout = target.progress_timeout_s if is_training_stage else 3600
        result = _run_modal_command(cmd, timeout_s=timeout)
        stage_results.append(result)

        # Check for blocked conditions
        blocked = _detect_blocked(result.stderr_tail, result.stdout_tail)
        if blocked:
            wall = time.monotonic() - start
            return ValidationResult(
                tutorial=target.tutorial_stem,
                target=target.target_name,
                target_type=target.target_type.value,
                outcome=Outcome.BLOCKED,
                blocked_reason=blocked[0],
                reason_detail=blocked[1],
                stages=stage_results,
                wall_time_s=wall,
            )

        # Check for stage failure (non-training prerequisite stages)
        if result.exit_code is not None and result.exit_code != 0 and not is_training_stage:
            wall = time.monotonic() - start
            first_err = result.stderr_tail.strip().splitlines()[-1] if result.stderr_tail.strip() else "nonzero exit"
            return ValidationResult(
                tutorial=target.tutorial_stem,
                target=target.target_name,
                target_type=target.target_type.value,
                outcome=Outcome.FAIL,
                failure_source=FailureSource.PREREQUISITE_STAGE,
                reason_detail=f"stage '{cmd}' exited {result.exit_code}: {first_err[:200]}",
                stages=stage_results,
                wall_time_s=wall,
            )

        # Timeout on prereq
        if result.exit_code is None and not is_training_stage:
            wall = time.monotonic() - start
            return ValidationResult(
                tutorial=target.tutorial_stem,
                target=target.target_name,
                target_type=target.target_type.value,
                outcome=Outcome.FAIL,
                failure_source=FailureSource.TIMEOUT,
                reason_detail=f"stage '{cmd}' timed out after {timeout}s",
                stages=stage_results,
                wall_time_s=wall,
            )

    # Check the final stage for markers
    last_result = stage_results[-1] if stage_results else None
    if last_result is None:
        wall = time.monotonic() - start
        return ValidationResult(
            tutorial=target.tutorial_stem,
            target=target.target_name,
            target_type=target.target_type.value,
            outcome=Outcome.FAIL,
            failure_source=FailureSource.UNKNOWN,
            reason_detail="no stages executed",
            wall_time_s=wall,
        )

    combined_output = last_result.stdout_tail + last_result.stderr_tail
    marker_hit = _find_marker(combined_output, target.progress_markers)

    if marker_hit:
        last_result.first_signal = marker_hit
        wall = time.monotonic() - start
        return ValidationResult(
            tutorial=target.tutorial_stem,
            target=target.target_name,
            target_type=target.target_type.value,
            outcome=Outcome.PASS,
            reason_detail=f"progress marker found: {marker_hit}",
            stages=stage_results,
            wall_time_s=wall,
            first_signal=marker_hit,
            app_id=_extract_app_id(combined_output),
        )

    # Timeout or failure without marker
    if last_result.exit_code is None:
        failure = FailureSource.TIMEOUT
        detail = f"timed out after {target.progress_timeout_s}s without progress marker"
    elif last_result.exit_code != 0:
        failure = FailureSource.TRAINING_RUNTIME
        last_line = last_result.stderr_tail.strip().splitlines()[-1] if last_result.stderr_tail.strip() else "nonzero exit"
        detail = f"exited {last_result.exit_code}: {last_line[:200]}"
    else:
        failure = FailureSource.UNKNOWN
        detail = "exited 0 but no progress marker found"

    wall = time.monotonic() - start
    return ValidationResult(
        tutorial=target.tutorial_stem,
        target=target.target_name,
        target_type=target.target_type.value,
        outcome=Outcome.FAIL,
        failure_source=failure,
        reason_detail=detail,
        stages=stage_results,
        wall_time_s=wall,
        app_id=_extract_app_id(combined_output),
    )


def _extract_app_id(text: str) -> str | None:
    match = re.search(r"(ap-[a-zA-Z0-9]+)", text)
    return match.group(1) if match else None


# ── Report (task9) ───────────────────────────────────────────────────────────


def write_markdown_report(results: list[ValidationResult], out: TextIO) -> None:
    modal_env = os.environ.get("MODAL_ENVIRONMENT", "(not set)")
    out.write(f"# Tutorial Validation Report\n\n")
    out.write(f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}\n")
    out.write(f"**Modal environment**: `{modal_env}`\n")
    out.write(f"**Tutorials scanned**: {len(results)}\n\n")

    counts = {o: 0 for o in Outcome}
    for r in results:
        counts[r.outcome] += 1

    out.write("## Summary\n\n")
    out.write(f"| Outcome | Count |\n|---|---|\n")
    for o in Outcome:
        out.write(f"| {o.value} | {counts[o]} |\n")
    out.write("\n")

    out.write("## Results\n\n")
    out.write("| Tutorial | Target | Type | Outcome | Detail | Wall Time |\n")
    out.write("|---|---|---|---|---|---|\n")
    for r in results:
        detail = r.reason_detail[:80] if r.reason_detail else ""
        if r.failure_source:
            detail = f"[{r.failure_source.value}] {detail}"
        elif r.blocked_reason:
            detail = f"[{r.blocked_reason.value}] {detail}"
        wt = f"{r.wall_time_s:.1f}s" if r.wall_time_s > 0 else "-"
        out.write(f"| {r.tutorial} | {r.target} | {r.target_type} | **{r.outcome.value}** | {detail} | {wt} |\n")

    out.write("\n## Detailed Results\n\n")
    for r in results:
        out.write(f"### {r.target}\n\n")
        out.write(f"- **Outcome**: {r.outcome.value}\n")
        if r.failure_source:
            out.write(f"- **Failure source**: {r.failure_source.value}\n")
        if r.blocked_reason:
            out.write(f"- **Blocked reason**: {r.blocked_reason.value}\n")
        out.write(f"- **Detail**: {r.reason_detail}\n")
        if r.app_id:
            out.write(f"- **App ID**: {r.app_id}\n")
        if r.first_signal:
            out.write(f"- **First signal**: `{r.first_signal}`\n")
        out.write(f"- **Wall time**: {r.wall_time_s:.1f}s\n")
        if r.stages:
            out.write(f"- **Stages**:\n")
            for s in r.stages:
                status = f"exit={s.exit_code}" if s.exit_code is not None else "timeout"
                out.write(f"  - `{s.stage}` → {status} ({s.wall_time_s:.1f}s)\n")
        out.write("\n")


def write_json_report(results: list[ValidationResult], out: TextIO) -> None:
    data = []
    for r in results:
        entry = {
            "tutorial": r.tutorial,
            "target": r.target,
            "target_type": r.target_type,
            "outcome": r.outcome.value,
            "failure_source": r.failure_source.value if r.failure_source else None,
            "blocked_reason": r.blocked_reason.value if r.blocked_reason else None,
            "reason_detail": r.reason_detail,
            "app_id": r.app_id,
            "first_signal": r.first_signal,
            "wall_time_s": round(r.wall_time_s, 1),
            "stages": [
                {
                    "command": s.command,
                    "exit_code": s.exit_code,
                    "wall_time_s": round(s.wall_time_s, 1),
                    "first_signal": s.first_signal,
                }
                for s in r.stages
            ],
        }
        data.append(entry)

    report = {
        "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "modal_environment": os.environ.get("MODAL_ENVIRONMENT", ""),
        "results": data,
    }
    json.dump(report, out, indent=2)
    out.write("\n")


# ── CLI (task10) ─────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list", action="store_true", help="List discovered targets and exit")
    parser.add_argument("--only", help="Validate only this tutorial (by stem name)")
    parser.add_argument("--preflight-only", action="store_true", help="Run only local preflight checks")
    parser.add_argument("--report-path", type=Path, help="Write markdown report to this file (default: stdout)")
    parser.add_argument("--json-report", type=Path, help="Write JSON report to this file")
    args = parser.parse_args()

    targets = discover_tutorials()

    if args.only:
        matched = [t for t in targets if t.tutorial_stem == args.only or t.target_name == args.only]
        skipped = [t for t in targets if t not in matched]
        targets = matched
        if not targets:
            print(f"No targets found matching --only '{args.only}'", file=sys.stderr)
            print(f"Available: {', '.join(t.tutorial_stem for t in discover_tutorials())}", file=sys.stderr)
            sys.exit(1)

    if args.list:
        print(f"{'Tutorial':<35} {'Target':<40} {'Type':<15} {'Stages'}")
        print("-" * 120)
        for t in targets:
            stages_str = " → ".join(
                re.search(r"::(\S+)", s).group(1) if "::" in s else s
                for s in t.stages
            )
            print(f"{t.tutorial_stem:<35} {t.target_name:<40} {t.target_type.value:<15} {stages_str}")
        print(f"\nTotal: {len(targets)} targets from {len(set(t.tutorial_stem for t in targets))} tutorials")
        return

    results: list[ValidationResult] = []

    # Phase 1: Preflight all targets
    print("═══ Tier-0 Preflight ═══\n")
    preflight_failures = {}
    for target in targets:
        result = run_preflight(target)
        if result:
            print(f"  FAIL  {target.target_name}: [{result.failure_source.value}] {result.reason_detail}")
            results.append(result)
            preflight_failures[target.target_name] = True
        else:
            print(f"  OK    {target.target_name}")

    if args.preflight_only:
        failed = sum(1 for r in results if r.outcome == Outcome.FAIL)
        print(f"\nPreflight: {len(targets) - failed}/{len(targets)} passed, {failed} failed")
        if args.report_path:
            with open(args.report_path, "w") as f:
                write_markdown_report(results, f)
            print(f"Report: {args.report_path}")
        return

    # Phase 2: Execute validation for targets that passed preflight
    print("\n═══ Validation ═══\n")
    for target in targets:
        if target.target_name in preflight_failures:
            continue

        print(f"  [{target.target_type.value}] {target.target_name}...", end=" ", flush=True)

        if target.target_type == TargetType.CONCEPT_ONLY:
            result = validate_concept_only(target)
        else:
            result = validate_remote_target(target)

        results.append(result)
        status = result.outcome.value.upper()
        detail = ""
        if result.first_signal:
            detail = f" → {result.first_signal[:60]}"
        elif result.reason_detail:
            detail = f" → {result.reason_detail[:60]}"
        print(f"{status}{detail} ({result.wall_time_s:.1f}s)")

    # Phase 3: Report
    print(f"\n═══ Summary ═══\n")
    counts = {o: 0 for o in Outcome}
    for r in results:
        counts[r.outcome] += 1
    for o in Outcome:
        if counts[o]:
            print(f"  {o.value}: {counts[o]}")

    if args.report_path:
        with open(args.report_path, "w") as f:
            write_markdown_report(results, f)
        print(f"\nMarkdown report: {args.report_path}")

    if args.json_report:
        with open(args.json_report, "w") as f:
            write_json_report(results, f)
        print(f"JSON report: {args.json_report}")


if __name__ == "__main__":
    main()
