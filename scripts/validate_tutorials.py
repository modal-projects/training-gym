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
import queue
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TextIO

REPO_ROOT = Path(__file__).resolve().parent.parent
GENERATOR_DIR = REPO_ROOT / "tutorials" / "tutorial_generator"
TUTORIALS_DIR = REPO_ROOT / "tutorials"

IMAGE_BUILD_TIMEOUT = 1800
CAPACITY_TIMEOUT = 900
PREREQ_TIMEOUT = 3600
CONVERT_TIMEOUT = 7200
UPLOAD_TIMEOUT = 120
DEFAULT_PROGRESS_TIMEOUT = 1800
BENCHMARK_TIMEOUT = 600
PATTERN_DEMO_TIMEOUT = 600
LAUNCH_TIMEOUT = 600


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


@dataclass
class CommandRecord:
    command: str
    entrypoint: str
    is_detached: bool


@dataclass
class TutorialTarget:
    tutorial_stem: str
    bucket: str
    source_path: Path
    generated_py: Path
    target_name: str
    target_type: TargetType
    prereq_commands: list[str]
    joy/initial-setup_command: str
    joy/initial-setup_is_detached: bool
    progress_markers: list[str]
    progress_timeout_s: int = DEFAULT_PROGRESS_TIMEOUT
    metadata: dict = field(default_factory=dict)


@dataclass
class StageResult:
    stage: str
    command: str
    exit_code: int | None = None
    app_id: str | None = None
    first_startup: str | None = None
    first_signal: str | None = None
    first_error: str | None = None
    classified_failure: FailureSource | None = None
    app_state: str | None = None
    wall_time_s: float = 0.0
    stdout_tail: str = ""
    stderr_tail: str = ""


@dataclass
class ValidationResult:
    tutorial: str
    target: str
    target_type: str
    outcome: Outcome
    command: str = ""
    failure_source: FailureSource | None = None
    blocked_reason: BlockedReason | None = None
    reason_detail: str = ""
    stages: list[StageResult] = field(default_factory=list)
    wall_time_s: float = 0.0
    app_id: str | None = None
    first_startup: str | None = None
    first_signal: str | None = None
    first_error: str | None = None


# ── Discovery ────────────────────────────────────────────────────────────────


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


def _extract_run_cli_commands(source: str) -> list[CommandRecord]:
    tree = ast.parse(source)
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if "_run_cli" not in node.name:
            continue
        doc = ast.get_docstring(node, clean=True) or ""
        records: list[CommandRecord] = []
        for line in doc.splitlines():
            stripped = line.strip()
            if "modal run" not in stripped or stripped.startswith("#"):
                continue
            if " -- " in stripped:
                continue
            if "--entrypoint" in stripped:
                continue
            is_detached = "--detach" in stripped or " -d " in stripped
            cleaned = stripped.removeprefix("uv run ").strip()
            match = re.search(r"::([\w.]+)", cleaned)
            if match:
                entrypoint = match.group(1)
                records.append(
                    CommandRecord(
                        command=cleaned,
                        entrypoint=entrypoint,
                        is_detached=is_detached,
                    )
                )
        return records
    return []


def _bucket_from_path(source_path: Path) -> str:
    rel = source_path.relative_to(GENERATOR_DIR)
    parts = rel.parts
    return parts[0] if len(parts) > 1 else ""


def _resolve_generated_py(bucket: str, stem: str) -> Path:
    if bucket:
        return TUTORIALS_DIR / bucket / stem / f"{stem}.py"
    return TUTORIALS_DIR / stem / f"{stem}.py"


def _entrypoint_prefix(ep: str) -> str:
    return ep.rsplit(".", 1)[0] if "." in ep else ""


_SHORTCUT_ENTRYPOINTS: dict[str, set[str]] = {}


def _derive_targets(
    stem: str,
    bucket: str,
    source_path: Path,
    generated_py: Path,
    records: list[CommandRecord],
    classification: dict,
    metadata: dict,
) -> list[TutorialTarget]:
    target_type: TargetType = classification["type"]
    markers: list[str] = classification["markers"]
    timeout: int = classification.get("timeout", DEFAULT_PROGRESS_TIMEOUT)
    train_suffix: str = classification.get("train_stage_suffix", "")

    if target_type == TargetType.CONCEPT_ONLY:
        return [
            TutorialTarget(
                tutorial_stem=stem,
                bucket=bucket,
                source_path=source_path,
                generated_py=generated_py,
                target_name=stem,
                target_type=target_type,
                prereq_commands=[],
                joy/initial-setup_command="",
                joy/initial-setup_is_detached=False,
                progress_markers=markers,
                progress_timeout_s=timeout,
                metadata=metadata,
            )
        ]

    seen: set[str] = set()
    deduped: list[CommandRecord] = []
    all_eps: set[str] = set()
    for r in records:
        if r.entrypoint not in seen:
            seen.add(r.entrypoint)
            all_eps.add(r.entrypoint)
            deduped.append(r)

    deduped = [
        r
        for r in deduped
        if r.entrypoint not in _SHORTCUT_ENTRYPOINTS
        or not _SHORTCUT_ENTRYPOINTS[r.entrypoint].issubset(all_eps)
    ]

    detached_indices = [i for i, r in enumerate(deduped) if r.is_detached]

    if not detached_indices:
        all_cmds = [r.command for r in deduped]
        joy/initial-setup_cmd = all_cmds[-1] if all_cmds else ""
        prereqs = all_cmds[:-1] if len(all_cmds) > 1 else []
        if train_suffix and joy/initial-setup_cmd:
            joy/initial-setup_cmd = f"{joy/initial-setup_cmd} {train_suffix}"
        return [
            TutorialTarget(
                tutorial_stem=stem,
                bucket=bucket,
                source_path=source_path,
                generated_py=generated_py,
                target_name=stem,
                target_type=target_type,
                prereq_commands=prereqs,
                joy/initial-setup_command=joy/initial-setup_cmd,
                joy/initial-setup_is_detached=False,
                progress_markers=markers,
                progress_timeout_s=timeout,
                metadata=metadata,
            )
        ]

    targets: list[TutorialTarget] = []
    for di in detached_indices:
        detached_rec = deduped[di]
        det_prefix = _entrypoint_prefix(detached_rec.entrypoint)

        prereqs: list[str] = []
        for r in deduped[:di]:
            if r.is_detached:
                continue
            r_prefix = _entrypoint_prefix(r.entrypoint)
            if not det_prefix or not r_prefix or r_prefix == det_prefix:
                prereqs.append(r.command)

        joy/initial-setup_cmd = detached_rec.command
        if train_suffix:
            joy/initial-setup_cmd = f"{joy/initial-setup_cmd} {train_suffix}"

        target_name = stem
        if len(detached_indices) > 1:
            short = detached_rec.entrypoint.split(".")[-1]
            ep_prefix = det_prefix.replace("_app", "") if det_prefix else short
            target_name = f"{stem}/{ep_prefix}"

        targets.append(
            TutorialTarget(
                tutorial_stem=stem,
                bucket=bucket,
                source_path=source_path,
                generated_py=generated_py,
                target_name=target_name,
                target_type=target_type,
                prereq_commands=prereqs,
                joy/initial-setup_command=joy/initial-setup_cmd,
                joy/initial-setup_is_detached=True,
                progress_markers=markers,
                progress_timeout_s=timeout,
                metadata=metadata,
            )
        )

    return targets


def discover_tutorials() -> list[TutorialTarget]:
    all_targets: list[TutorialTarget] = []
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
        classification = TUTORIAL_CLASSIFICATIONS.get(stem)
        if classification is None:
            continue
        generated_py = _resolve_generated_py(bucket, stem)
        records = _extract_run_cli_commands(source)
        targets = _derive_targets(
            stem, bucket, source_path, generated_py, records, classification, metadata
        )
        all_targets.extend(targets)
    return all_targets


TUTORIAL_CLASSIFICATIONS: dict[str, dict] = {
    "quickstart": {"type": TargetType.CONCEPT_ONLY, "markers": []},
    "nccl_benchmark": {
        "type": TargetType.BENCHMARK,
        "markers": ["busbw:", "algbw:"],
        "timeout": BENCHMARK_TIMEOUT,
    },
    "ray_slime_standalone": {
        "type": TargetType.PATTERN_DEMO,
        "markers": ["Ray dashboard:", "ray.cluster_resources"],
        "timeout": PATTERN_DEMO_TIMEOUT,
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


# ── Preflight ────────────────────────────────────────────────────────────────


def run_preflight(target: TutorialTarget) -> ValidationResult | None:
    try:
        source = target.source_path.read_text()
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
                reason_detail=f"{target.source_path} newer than {target.generated_py}. "
                "Regenerate via: uv run python tutorials/generate_tutorial.py",
            )
    elif target.target_type != TargetType.CONCEPT_ONLY:
        return ValidationResult(
            tutorial=target.tutorial_stem,
            target=target.target_name,
            target_type=target.target_type.value,
            outcome=Outcome.FAIL,
            failure_source=FailureSource.GENERATOR_DRIFT,
            reason_detail=f"generated file {target.generated_py} does not exist",
        )

    if target.target_type != TargetType.CONCEPT_ONLY and target.generated_py.exists():
        all_commands = target.prereq_commands + (
            [target.joy/initial-setup_command] if target.joy/initial-setup_command else []
        )
        for cmd in all_commands:
            match = re.search(r"::([\w.]+)", cmd)
            if match:
                entrypoint = match.group(1)
                if not _entrypoint_exists_in_module(target.generated_py, entrypoint):
                    return ValidationResult(
                        tutorial=target.tutorial_stem,
                        target=target.target_name,
                        target_type=target.target_type.value,
                        outcome=Outcome.FAIL,
                        failure_source=FailureSource.ARG_PARSE,
                        reason_detail=f"entrypoint '{entrypoint}' not defined as a callable in {target.generated_py}",
                    )
    return None


def _entrypoint_exists_in_module(gen_py: Path, entrypoint: str) -> bool:
    """Verify entrypoint exists as a callable, not just in comments.

    Two patterns:
    1. Standalone tutorials: function defined directly (e.g. @app.function def run_benchmark)
    2. Framework tutorials: app created via build_app() factory — the framework launcher
       defines entrypoints dynamically, so we verify the factory call exists instead.
    """
    func_name = entrypoint.split(".")[-1]
    try:
        tree = ast.parse(gen_py.read_text())
    except SyntaxError:
        return False

    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == func_name
        ):
            return True

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "build_app":
                return True

    return False


# ── Executors ────────────────────────────────────────────────────────────────


def _extract_app_id(text: str) -> str | None:
    match = re.search(r"(ap-[a-zA-Z0-9]+)", text)
    return match.group(1) if match else None


def _find_marker_in_line(line: str, markers: list[str]) -> str | None:
    low = line.lower()
    for m in markers:
        if m.lower() in low:
            return line.strip()[:200]
    return None


_STARTUP_PATTERNS = [
    "rank",
    "hello from",
    "container",
    "node_rank",
    "Starting",
    "Initialized",
    "worker",
]


def _is_startup_line(line: str) -> bool:
    low = line.lower()
    return any(p.lower() in low for p in _STARTUP_PATTERNS)


def _detect_blocked_in_text(text: str) -> tuple[BlockedReason, str] | None:
    if re.search(r"40[13].*(?:huggingface|hf\.co)", text, re.IGNORECASE):
        return BlockedReason.GATED_ACCESS, "HF 401/403 — gated model/dataset access"
    if re.search(r"(?:huggingface|hf\.co).*40[13]", text, re.IGNORECASE):
        return BlockedReason.GATED_ACCESS, "HF 401/403 — gated model/dataset access"
    m = re.search(
        r"[Ss]ecret\s+['\"]?(\S+?)['\"]?\s+(?:not found|does not exist)", text
    )
    if m:
        return BlockedReason.MISSING_SECRET, f"missing Modal secret: {m.group(1)}"
    if "no capacity" in text.lower() or "insufficient capacity" in text.lower():
        return BlockedReason.CAPACITY, "cluster capacity unavailable"
    return None


def _run_attached(cmd: str, timeout_s: int) -> StageResult:
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
        stderr_lines = proc.stderr.strip().splitlines() if proc.stderr else []
        first_err = (
            stderr_lines[-1][:200] if stderr_lines and proc.returncode != 0 else None
        )
        return StageResult(
            stage=cmd,
            command=full_cmd,
            exit_code=proc.returncode,
            app_id=_extract_app_id(proc.stdout + proc.stderr),
            first_startup=None,
            first_signal=None,
            first_error=first_err,
            wall_time_s=wall,
            stdout_tail=proc.stdout[-2000:] if proc.stdout else "",
            stderr_tail=proc.stderr[-2000:] if proc.stderr else "",
        )
    except subprocess.TimeoutExpired:
        return StageResult(
            stage=cmd,
            command=full_cmd,
            exit_code=None,
            wall_time_s=time.monotonic() - start,
            first_error="timeout",
        )


def _poll_app_state(app_id: str) -> str | None:
    """Poll modal app list --json for the given app's lifecycle state."""
    try:
        result = subprocess.run(
            ["uv", "run", "modal", "app", "list", "--json"],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(REPO_ROOT),
        )
        if result.returncode != 0:
            return None
        apps = json.loads(result.stdout)
        for app in apps:
            aid = app.get("app_id", "") or app.get("App ID", "") or ""
            if app_id in aid or app_id in json.dumps(app):
                return (app.get("state") or app.get("State") or "unknown").lower()
    except Exception:
        pass
    return None


def _classify_detached_failure(
    app_id: str | None,
    app_state: str | None,
    t1_reached: bool,
    first_startup: str | None,
    first_error: str | None,
    combined_text: str,
    elapsed_since_launch: float,
) -> tuple[FailureSource, str]:
    """Apply T0→T3 precedence-based classification for a detached stage failure."""
    blocked = _detect_blocked_in_text(combined_text)
    if blocked:
        if blocked[0] == BlockedReason.GATED_ACCESS:
            return FailureSource.GATED_ACCESS, blocked[1]
        if blocked[0] == BlockedReason.MISSING_SECRET:
            return FailureSource.SECRET_MISSING, blocked[1]
        if blocked[0] == BlockedReason.CAPACITY:
            return FailureSource.CAPACITY, blocked[1]

    if not t1_reached:
        if app_state and "creating" in app_state:
            return (
                FailureSource.IMAGE_BUILD,
                f"app stuck in '{app_state}' after {elapsed_since_launch:.0f}s",
            )
        if app_state and ("queue" in app_state or "waiting" in app_state):
            return (
                FailureSource.CAPACITY,
                f"app stuck in '{app_state}' — capacity unavailable",
            )
        return FailureSource.TIMEOUT, "app never reached running state"

    if not first_startup:
        return FailureSource.TIMEOUT, "app running but never reached user code"

    return (
        FailureSource.TIMEOUT,
        "user code started but no progress marker within timeout",
    )


def _run_detached(cmd: str, markers: list[str], progress_timeout_s: int) -> StageResult:
    full_cmd = f"uv run {cmd}"
    start = time.monotonic()

    try:
        launch = subprocess.run(
            full_cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=LAUNCH_TIMEOUT,
            cwd=str(REPO_ROOT),
        )
    except subprocess.TimeoutExpired:
        return StageResult(
            stage=cmd,
            command=full_cmd,
            exit_code=None,
            first_error=f"detached launch timed out after {LAUNCH_TIMEOUT}s",
            classified_failure=FailureSource.IMAGE_BUILD,
            wall_time_s=time.monotonic() - start,
        )
    launch_output = launch.stdout + launch.stderr
    app_id = _extract_app_id(launch_output)

    if launch.returncode != 0 or not app_id:
        blocked = _detect_blocked_in_text(launch_output)
        if blocked:
            return StageResult(
                stage=cmd,
                command=full_cmd,
                exit_code=launch.returncode,
                first_error=launch.stderr.strip()[-200:] if launch.stderr else None,
                wall_time_s=time.monotonic() - start,
                stdout_tail=launch.stdout[-2000:],
                stderr_tail=launch.stderr[-2000:],
            )
        return StageResult(
            stage=cmd,
            command=full_cmd,
            exit_code=launch.returncode,
            first_error=launch.stderr.strip()[-200:]
            if launch.stderr
            else "launch failed",
            wall_time_s=time.monotonic() - start,
            stdout_tail=launch.stdout[-2000:],
            stderr_tail=launch.stderr[-2000:],
        )

    first_startup = None
    first_signal = None
    first_error = None
    log_lines: list[str] = []
    t1_reached = False
    app_state: str | None = None
    last_poll = time.monotonic()

    logs_cmd = f"uv run modal app logs {app_id}"
    logs_proc = subprocess.Popen(
        logs_cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=str(REPO_ROOT),
    )
    line_queue: queue.Queue[str | None] = queue.Queue()

    def _reader():
        assert logs_proc.stdout is not None
        for ln in logs_proc.stdout:
            line_queue.put(ln)
        line_queue.put(None)

    reader_thread = threading.Thread(target=_reader, daemon=True)
    reader_thread.start()

    overall_timeout = progress_timeout_s + IMAGE_BUILD_TIMEOUT + CAPACITY_TIMEOUT
    deadline = start + overall_timeout

    while time.monotonic() < deadline:
        try:
            line = line_queue.get(timeout=10)
        except queue.Empty:
            if logs_proc.poll() is not None:
                break
            if time.monotonic() - last_poll > 30:
                app_state = _poll_app_state(app_id)
                last_poll = time.monotonic()
                if app_state and app_state in ("running", "deployed", "ephemeral"):
                    t1_reached = True
                if app_state and not t1_reached:
                    elapsed = time.monotonic() - start
                    if (
                        "creating" in (app_state or "")
                        and elapsed > IMAGE_BUILD_TIMEOUT
                    ):
                        first_error = (
                            f"image build timeout ({elapsed:.0f}s in '{app_state}')"
                        )
                        break
                    if (
                        "queue" in (app_state or "") or "waiting" in (app_state or "")
                    ) and elapsed > CAPACITY_TIMEOUT:
                        first_error = (
                            f"capacity timeout ({elapsed:.0f}s in '{app_state}')"
                        )
                        break
            continue
        if line is None:
            break

        log_lines.append(line)
        if not t1_reached:
            t1_reached = True

        if not first_startup and _is_startup_line(line):
            first_startup = line.strip()[:200]
        if not first_signal:
            hit = _find_marker_in_line(line, markers)
            if hit:
                first_signal = hit
                break
        blocked = _detect_blocked_in_text(line)
        if blocked:
            first_error = f"[{blocked[0].value}] {blocked[1]}"
            break
        if not first_error:
            low = line.lower()
            if "error" in low or "traceback" in low or "exception" in low:
                first_error = line.strip()[:200]

    logs_proc.terminate()
    try:
        logs_proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        logs_proc.kill()

    if first_signal:
        try:
            subprocess.run(
                f"uv run modal app stop {app_id}",
                shell=True,
                capture_output=True,
                timeout=30,
                cwd=str(REPO_ROOT),
            )
        except Exception:
            pass

    wall = time.monotonic() - start
    tail = "".join(log_lines[-50:])

    classified = None
    if not first_signal:
        classified, detail = _classify_detached_failure(
            app_id,
            app_state,
            t1_reached,
            first_startup,
            first_error,
            tail,
            wall,
        )
        if not first_error:
            first_error = detail

    return StageResult(
        stage=cmd,
        command=full_cmd,
        exit_code=0 if first_signal else None,
        classified_failure=classified,
        app_state=app_state,
        app_id=app_id,
        first_startup=first_startup,
        first_signal=first_signal,
        first_error=first_error,
        wall_time_s=wall,
        stdout_tail=tail,
        stderr_tail="",
    )


def _prereq_timeout(cmd: str) -> int:
    low = cmd.lower()
    if "convert" in low:
        return CONVERT_TIMEOUT
    if "upload" in low:
        return UPLOAD_TIMEOUT
    return PREREQ_TIMEOUT


def validate_concept_only(target: TutorialTarget) -> ValidationResult:
    start = time.monotonic()
    try:
        gen_source = target.generated_py.read_text()
        compile(gen_source, str(target.generated_py), "exec")
    except Exception as e:
        return ValidationResult(
            tutorial=target.tutorial_stem,
            target=target.target_name,
            target_type=target.target_type.value,
            outcome=Outcome.FAIL,
            failure_source=FailureSource.LOCAL_COMPILE,
            reason_detail=str(e),
            wall_time_s=time.monotonic() - start,
        )

    import_check = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "-c",
            f"import importlib.util; spec = importlib.util.spec_from_file_location('test', '{target.generated_py}'); "
            f"mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)",
        ],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=str(REPO_ROOT),
    )
    wall = time.monotonic() - start
    if import_check.returncode != 0:
        return ValidationResult(
            tutorial=target.tutorial_stem,
            target=target.target_name,
            target_type=target.target_type.value,
            outcome=Outcome.FAIL,
            failure_source=FailureSource.LOCAL_COMPILE,
            reason_detail=f"import failed: {import_check.stderr.strip()[-300:]}",
            wall_time_s=wall,
        )
    return ValidationResult(
        tutorial=target.tutorial_stem,
        target=target.target_name,
        target_type=target.target_type.value,
        outcome=Outcome.PASS,
        reason_detail="source compiles and imports resolve",
        wall_time_s=wall,
        first_signal="import_ok",
    )


def validate_remote_target(target: TutorialTarget) -> ValidationResult:
    start = time.monotonic()
    stage_results: list[StageResult] = []

    for cmd in target.prereq_commands:
        timeout = _prereq_timeout(cmd)
        print(f"\n    prereq: {cmd}", flush=True)
        result = _run_attached(cmd, timeout_s=timeout)
        stage_results.append(result)

        blocked = _detect_blocked_in_text(result.stdout_tail + result.stderr_tail)
        if blocked:
            return ValidationResult(
                tutorial=target.tutorial_stem,
                target=target.target_name,
                target_type=target.target_type.value,
                outcome=Outcome.BLOCKED,
                command=cmd,
                blocked_reason=blocked[0],
                reason_detail=blocked[1],
                stages=stage_results,
                wall_time_s=time.monotonic() - start,
            )
        if result.exit_code is not None and result.exit_code != 0:
            return ValidationResult(
                tutorial=target.tutorial_stem,
                target=target.target_name,
                target_type=target.target_type.value,
                outcome=Outcome.FAIL,
                command=cmd,
                failure_source=FailureSource.PREREQUISITE_STAGE,
                reason_detail=f"prereq '{cmd}' exited {result.exit_code}: {result.first_error or 'unknown'}",
                stages=stage_results,
                wall_time_s=time.monotonic() - start,
                first_error=result.first_error,
            )
        if result.exit_code is None:
            return ValidationResult(
                tutorial=target.tutorial_stem,
                target=target.target_name,
                target_type=target.target_type.value,
                outcome=Outcome.FAIL,
                command=cmd,
                failure_source=FailureSource.TIMEOUT,
                reason_detail=f"prereq '{cmd}' timed out after {timeout}s",
                stages=stage_results,
                wall_time_s=time.monotonic() - start,
            )

    if not target.joy/initial-setup_command:
        return ValidationResult(
            tutorial=target.tutorial_stem,
            target=target.target_name,
            target_type=target.target_type.value,
            outcome=Outcome.FAIL,
            failure_source=FailureSource.UNKNOWN,
            reason_detail="no main command",
            stages=stage_results,
            wall_time_s=time.monotonic() - start,
        )

    print(
        f"\n    main: {target.joy/initial-setup_command} {'(detached)' if target.joy/initial-setup_is_detached else '(attached)'}",
        flush=True,
    )

    if target.joy/initial-setup_is_detached:
        joy/initial-setup_result = _run_detached(
            target.joy/initial-setup_command, target.progress_markers, target.progress_timeout_s
        )
    else:
        joy/initial-setup_result = _run_attached(
            target.joy/initial-setup_command, timeout_s=target.progress_timeout_s
        )

    stage_results.append(joy/initial-setup_result)
    wall = time.monotonic() - start

    blocked = _detect_blocked_in_text(joy/initial-setup_result.stdout_tail + joy/initial-setup_result.stderr_tail)
    if blocked:
        return ValidationResult(
            tutorial=target.tutorial_stem,
            target=target.target_name,
            target_type=target.target_type.value,
            outcome=Outcome.BLOCKED,
            command=target.joy/initial-setup_command,
            blocked_reason=blocked[0],
            reason_detail=blocked[1],
            stages=stage_results,
            wall_time_s=wall,
            app_id=joy/initial-setup_result.app_id,
            first_error=joy/initial-setup_result.first_error,
        )

    if joy/initial-setup_result.first_signal:
        return ValidationResult(
            tutorial=target.tutorial_stem,
            target=target.target_name,
            target_type=target.target_type.value,
            outcome=Outcome.PASS,
            command=target.joy/initial-setup_command,
            reason_detail=f"progress marker: {joy/initial-setup_result.first_signal}",
            stages=stage_results,
            wall_time_s=wall,
            app_id=joy/initial-setup_result.app_id,
            first_startup=joy/initial-setup_result.first_startup,
            first_signal=joy/initial-setup_result.first_signal,
        )

    combined = joy/initial-setup_result.stdout_tail + joy/initial-setup_result.stderr_tail
    marker_from_attached = (
        _find_marker_in_line(combined, target.progress_markers)
        if not target.joy/initial-setup_is_detached
        else None
    )
    if marker_from_attached:
        return ValidationResult(
            tutorial=target.tutorial_stem,
            target=target.target_name,
            target_type=target.target_type.value,
            outcome=Outcome.PASS,
            command=target.joy/initial-setup_command,
            reason_detail=f"progress marker: {marker_from_attached}",
            stages=stage_results,
            wall_time_s=wall,
            app_id=joy/initial-setup_result.app_id,
            first_signal=marker_from_attached,
        )

    if joy/initial-setup_result.classified_failure:
        fs = joy/initial-setup_result.classified_failure
        detail = joy/initial-setup_result.first_error or f"classified as {fs.value}"
        if fs in (
            FailureSource.GATED_ACCESS,
            FailureSource.SECRET_MISSING,
            FailureSource.CAPACITY,
        ):
            blocked_map = {
                FailureSource.GATED_ACCESS: BlockedReason.GATED_ACCESS,
                FailureSource.SECRET_MISSING: BlockedReason.MISSING_SECRET,
                FailureSource.CAPACITY: BlockedReason.CAPACITY,
            }
            return ValidationResult(
                tutorial=target.tutorial_stem,
                target=target.target_name,
                target_type=target.target_type.value,
                outcome=Outcome.BLOCKED,
                command=target.joy/initial-setup_command,
                blocked_reason=blocked_map[fs],
                reason_detail=detail,
                stages=stage_results,
                wall_time_s=wall,
                app_id=joy/initial-setup_result.app_id,
                first_error=joy/initial-setup_result.first_error,
            )
    elif joy/initial-setup_result.exit_code is None:
        fs = FailureSource.TIMEOUT
        detail = f"timed out after {target.progress_timeout_s}s without progress marker"
    elif joy/initial-setup_result.exit_code != 0:
        fs = FailureSource.TRAINING_RUNTIME
        detail = (
            f"exited {joy/initial-setup_result.exit_code}: {joy/initial-setup_result.first_error or 'unknown'}"
        )
    else:
        fs = FailureSource.UNKNOWN
        detail = "exited 0 but no progress marker found"

    return ValidationResult(
        tutorial=target.tutorial_stem,
        target=target.target_name,
        target_type=target.target_type.value,
        outcome=Outcome.FAIL,
        command=target.joy/initial-setup_command,
        failure_source=fs,
        reason_detail=detail,
        stages=stage_results,
        wall_time_s=wall,
        app_id=joy/initial-setup_result.app_id,
        first_startup=joy/initial-setup_result.first_startup,
        first_error=joy/initial-setup_result.first_error,
    )


# ── Report ───────────────────────────────────────────────────────────────────


def write_markdown_report(results: list[ValidationResult], out: TextIO) -> None:
    modal_env = os.environ.get("MODAL_ENVIRONMENT", "(not set)")
    out.write("# Tutorial Validation Report\n\n")
    out.write(f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}\n")
    out.write(f"**Modal environment**: `{modal_env}`\n")
    out.write(f"**Targets scanned**: {len(results)}\n\n")

    counts = {o: 0 for o in Outcome}
    for r in results:
        counts[r.outcome] += 1

    out.write("## Summary\n\n| Outcome | Count |\n|---|---|\n")
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
        out.write(
            f"| {r.tutorial} | {r.target} | {r.target_type} | **{r.outcome.value}** | {detail} | {wt} |\n"
        )

    out.write("\n## Detailed Results\n\n")
    for r in results:
        out.write(f"### {r.target}\n\n")
        out.write(f"- **Outcome**: {r.outcome.value}\n")
        out.write(f"- **Command**: `{r.command}`\n")
        if r.failure_source:
            out.write(f"- **Failure source**: {r.failure_source.value}\n")
        if r.blocked_reason:
            out.write(f"- **Blocked reason**: {r.blocked_reason.value}\n")
        out.write(f"- **Detail**: {r.reason_detail}\n")
        if r.app_id:
            out.write(f"- **App ID**: `{r.app_id}`\n")
        if r.first_startup:
            out.write(f"- **First startup**: `{r.first_startup}`\n")
        if r.first_signal:
            out.write(f"- **First signal**: `{r.first_signal}`\n")
        if r.first_error:
            out.write(f"- **First error**: `{r.first_error}`\n")
        out.write(f"- **Wall time**: {r.wall_time_s:.1f}s\n")
        if r.stages:
            out.write("- **Stages**:\n")
            for s in r.stages:
                status = f"exit={s.exit_code}" if s.exit_code is not None else "timeout"
                app = f" app={s.app_id}" if s.app_id else ""
                state = f" state={s.app_state}" if s.app_state else ""
                classified = (
                    f" [{s.classified_failure.value}]" if s.classified_failure else ""
                )
                out.write(
                    f"  - `{s.stage}` → {status}{app}{state}{classified} ({s.wall_time_s:.1f}s)\n"
                )
                if s.first_startup:
                    out.write(f"    startup: `{s.first_startup}`\n")
                if s.first_signal:
                    out.write(f"    signal: `{s.first_signal}`\n")
                if s.first_error:
                    out.write(f"    error: `{s.first_error}`\n")
        out.write("\n")


def write_json_report(results: list[ValidationResult], out: TextIO) -> None:
    data = []
    for r in results:
        data.append(
            {
                "tutorial": r.tutorial,
                "target": r.target,
                "target_type": r.target_type,
                "outcome": r.outcome.value,
                "command": r.command,
                "failure_source": r.failure_source.value if r.failure_source else None,
                "blocked_reason": r.blocked_reason.value if r.blocked_reason else None,
                "reason_detail": r.reason_detail,
                "app_id": r.app_id,
                "first_startup": r.first_startup,
                "first_signal": r.first_signal,
                "first_error": r.first_error,
                "wall_time_s": round(r.wall_time_s, 1),
                "stages": [
                    {
                        "stage": s.stage,
                        "command": s.command,
                        "exit_code": s.exit_code,
                        "app_id": s.app_id,
                        "app_state": s.app_state,
                        "classified_failure": s.classified_failure.value
                        if s.classified_failure
                        else None,
                        "first_startup": s.first_startup,
                        "first_signal": s.first_signal,
                        "first_error": s.first_error,
                        "wall_time_s": round(s.wall_time_s, 1),
                    }
                    for s in r.stages
                ],
            }
        )
    json.dump(
        {
            "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "modal_environment": os.environ.get("MODAL_ENVIRONMENT", ""),
            "results": data,
        },
        out,
        indent=2,
    )
    out.write("\n")


# ── CLI ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--list", action="store_true", help="List discovered targets and exit"
    )
    parser.add_argument("--only", help="Validate only this tutorial (by stem name)")
    parser.add_argument(
        "--preflight-only", action="store_true", help="Run only local preflight checks"
    )
    parser.add_argument(
        "--report-path", type=Path, help="Write markdown report to file"
    )
    parser.add_argument("--json-report", type=Path, help="Write JSON report to file")
    args = parser.parse_args()

    all_targets = discover_tutorials()

    if args.list:
        print(f"{'Tutorial':<35} {'Target':<40} {'Type':<15} {'Stages'}")
        print("-" * 130)
        for t in all_targets:
            prereq_eps = [
                re.search(r"::([\w.]+)", c).group(1) if "::" in c else c
                for c in t.prereq_commands
            ]
            joy/initial-setup_ep = (
                re.search(r"::([\w.]+)", t.joy/initial-setup_command).group(1)
                if t.joy/initial-setup_command and "::" in t.joy/initial-setup_command
                else t.joy/initial-setup_command or "(local)"
            )
            det = " [detached]" if t.joy/initial-setup_is_detached else ""
            stages_str = " → ".join(prereq_eps + [f"{joy/initial-setup_ep}{det}"])
            print(
                f"{t.tutorial_stem:<35} {t.target_name:<40} {t.target_type.value:<15} {stages_str}"
            )
        print(
            f"\nTotal: {len(all_targets)} targets from {len(set(t.tutorial_stem for t in all_targets))} tutorials"
        )
        return

    skipped_results: list[ValidationResult] = []
    if args.only:
        matched = [
            t
            for t in all_targets
            if t.tutorial_stem == args.only or t.target_name == args.only
        ]
        excluded = [t for t in all_targets if t not in matched]
        for t in excluded:
            skipped_results.append(
                ValidationResult(
                    tutorial=t.tutorial_stem,
                    target=t.target_name,
                    target_type=t.target_type.value,
                    outcome=Outcome.SKIPPED,
                    reason_detail="filter_excluded",
                )
            )
        all_targets = matched
        if not all_targets:
            print(f"No targets found matching --only '{args.only}'", file=sys.stderr)
            sys.exit(1)

    results: list[ValidationResult] = []

    print("═══ Tier-0 Preflight ═══\n")
    preflight_failures: set[str] = set()
    for target in all_targets:
        result = run_preflight(target)
        if result:
            print(
                f"  FAIL  {target.target_name}: [{result.failure_source.value}] {result.reason_detail}"
            )
            results.append(result)
            preflight_failures.add(target.target_name)
        else:
            print(f"  OK    {target.target_name}")
            if args.preflight_only:
                results.append(
                    ValidationResult(
                        tutorial=target.tutorial_stem,
                        target=target.target_name,
                        target_type=target.target_type.value,
                        outcome=Outcome.PASS,
                        reason_detail="preflight passed",
                    )
                )

    if args.preflight_only:
        all_results = results + skipped_results
        failed = sum(1 for r in results if r.outcome == Outcome.FAIL)
        passed = sum(1 for r in results if r.outcome == Outcome.PASS)
        print(f"\nPreflight: {passed}/{len(all_targets)} passed, {failed} failed")
        _write_reports(all_results, args)
        return

    print("\n═══ Validation ═══\n")
    for target in all_targets:
        if target.target_name in preflight_failures:
            continue
        print(
            f"  [{target.target_type.value}] {target.target_name}...",
            end="",
            flush=True,
        )
        if target.target_type == TargetType.CONCEPT_ONLY:
            result = validate_concept_only(target)
        else:
            result = validate_remote_target(target)
        results.append(result)
        status = result.outcome.value.upper()
        sig = f" → {result.first_signal[:60]}" if result.first_signal else ""
        detail = (
            f" → {result.reason_detail[:60]}"
            if not sig and result.reason_detail
            else ""
        )
        print(f" {status}{sig}{detail} ({result.wall_time_s:.1f}s)")

    all_results = results + skipped_results
    print("\n═══ Summary ═══\n")
    counts = {o: 0 for o in Outcome}
    for r in all_results:
        counts[r.outcome] += 1
    for o in Outcome:
        if counts[o]:
            print(f"  {o.value}: {counts[o]}")

    _write_reports(all_results, args)


def _write_reports(results: list[ValidationResult], args: argparse.Namespace) -> None:
    if args.report_path:
        with open(args.report_path, "w") as f:
            write_markdown_report(results, f)
        print(f"\nMarkdown report: {args.report_path}")
    if args.json_report:
        with open(args.json_report, "w") as f:
            write_json_report(results, f)
        print(f"JSON report: {args.json_report}")


if __name__ == "__joy/initial-setup__":
    main()
