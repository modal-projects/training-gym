"""Learning Agent integrity lock — pin the fixed-benchmark surface, refuse to score on drift.

A benchmark number is only meaningful if the measurement instrument was identical
across runs. This module hashes everything that defines the instrument:

  - the harness code that produces and scores candidates
      harness/eval.py, harness/judge.py, harness/judge_cli.py, harness/grader.py,
      harness/integrity.py (yes, this file), bench.py
      Self-hashing is just hashing bytes on disk — well-defined, no recursion; an
      edit to the lock is benchmark drift and requires a deliberate re-freeze.
      Because a TAMPERED integrity.py could simply lie about itself, the pinned
      entrypoints (harness/judge_cli.py, bench.py) also hash this file directly
      against bench/pins.json before trusting verify_pins (`_verify_verifier`).
  - the spec               bench/config.yaml (global) + task_configs/<T>.yaml (per task)
  - per-task fixed assets  tasks/<T>/sys.txt, tasks/<T>/dev.json,
                           tasks/<T>/test.json, tasks/<T>/task.md, tasks/<T>/brief.md
  - secondary-metric data  each task's `api_surface` / `symbols` file from config
  - the corpora            content tree-hash of each task's corpus dir (volatile
                           entries like .git/__pycache__ excluded); a corpus dir
                           absent from a partial working copy is skipped at verify
                           (eval cannot run without it, so drift is never silent)
  - the judge prompt       sha256 of judge.build_judge_prompt's SOURCE (so a prompt
                           edit is caught even if judge.py's other bytes also moved)
  - the pinned eval budget value from config

`compute_pins()` builds the pin dict, `write_pins()` freezes it to bench/pins.json,
`verify_pins()` returns a list of human-readable mismatches (empty == clean).
bench.py `freeze`/`verify` and the judge's pre-scoring gate call these.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PINS_VERSION = 1

# Harness files under the lock — INCLUDING integrity.py itself and bench.py.
# A neutered copy of this file can still lie about itself, which is why the pinned
# entrypoints (judge_cli.py, bench.py) independently hash harness/integrity.py
# against bench/pins.json before trusting verify_pins.
HARNESS_FILES = [
    "harness/eval.py",
    "harness/judge.py",
    "harness/judge_cli.py",
    "harness/grader.py",
    "harness/integrity.py",
    "harness/envfile.py",
    "harness/config.py",
    "harness/rollout.py",
    "harness/rollout_modal.py",   # env image pins + the serving command: the
                                  #   vLLM version and --tool-call-parser decide
                                  #   whether a tool-calling policy works at all
    "harness/user_sim.py",        # the pinned customer simulator for
                                  #   conversational tasks IS environment
    "harness/judge_service.py",   # the judge service pins the judge model
                                  #   server-side — it IS the instrument
    "harness/adapters/__init__.py",
    "bench.py",
    "bench/config.yaml",
    # the contestant instruction IS part of the benchmark surface: the
    # template plus every block it can be stitched from (see
    # workspace_setup/setup_agent_md.py and instructions/README.md)
    "workspace_setup/instructions/AGENTS.md",
    "workspace_setup/instructions/objective/qa.md",
    "workspace_setup/instructions/objective/agentic.md",
    "workspace_setup/instructions/data_access/easy.md",
    "workspace_setup/instructions/data_access/medium.md",
    "workspace_setup/instructions/data_access/hard.md",
    "workspace_setup/instructions/setup/modal.md",
    "workspace_setup/instructions/methods/sft.md",
    "workspace_setup/instructions/methods/dpo.md",
    "workspace_setup/instructions/methods/rl.md",
    "workspace_setup/instructions/methods/opd.md",
    "workspace_setup/instructions/harness/qa.md",
    "workspace_setup/instructions/harness/agentic.md",
    "workspace_setup/instructions/tips/qa.md",
    "workspace_setup/instructions/tips/agentic.md",
    "workspace_setup/instructions/rules/default.md",
]
TASK_FILES = ["sys.txt", "dev.json", "test.json", "task.md", "brief.md"]
# Per-task config keys naming pinned DATA files (secondary-metric inputs, and
# the env adapter an agentic task scores through).
TASK_DATA_KEYS = ["api_surface", "symbols", "refs", "adapter"]
# Per-task config key naming the corpus directory (pinned as a content tree hash).
TASK_CORPUS_KEY = "corpus"
# Volatile/derived entries excluded from the corpus tree hash.
CORPUS_EXCLUDE_NAMES = {".git", "__pycache__", ".DS_Store"}
CORPUS_EXCLUDE_SUFFIXES = (".pyc",)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_tree(dirpath: Path) -> str:
    """Content hash of a directory tree: sorted relative paths + per-file sha256.
    Volatile/derived entries (.git, __pycache__, *.pyc, .DS_Store) are excluded so
    browsing/importing the corpus never reads as benchmark drift."""
    h = hashlib.sha256()
    entries = sorted(dirpath.rglob("*"), key=lambda p: p.relative_to(dirpath).as_posix())
    for p in entries:
        rel = p.relative_to(dirpath)
        if any(part in CORPUS_EXCLUDE_NAMES for part in rel.parts):
            continue
        if p.is_symlink() or not p.is_file() or p.name.endswith(CORPUS_EXCLUDE_SUFFIXES):
            continue
        h.update(rel.as_posix().encode())
        h.update(b"\0")
        h.update(_sha256_file(p).encode())
        h.update(b"\n")
    return h.hexdigest()


def _load_config(root: Path) -> dict:
    """Combined config view {"global": …, "tasks": {…}} via ROOT's own
    harness/config.py, loaded from the explicit file path (NOT `import config`):
    a plain import would return whatever module is cached in sys.modules —
    typically the real tree's — so verifying a copied/foreign root would read
    the wrong loader (same reasoning as judge_prompt_sha below)."""
    import importlib.util
    path = root / "harness" / "config.py"
    spec = importlib.util.spec_from_file_location(
        f"_cfg_{hashlib.sha256(str(path).encode()).hexdigest()[:12]}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.load_config(root)


def judge_prompt_sha(root: Path) -> str:
    """Hash of the judge-prompt TEMPLATE: the source of build_judge_prompt in
    ROOT's harness/judge.py.

    Loaded from the explicit file path (NOT `import judge`): a plain import would
    return whatever `judge` module is already cached in sys.modules — typically the
    real tree's — so verifying a copied/foreign root would silently hash the wrong
    source and a tampered prompt template would pass the pin check.
    """
    import importlib.util
    import linecache
    path = root / "harness" / "judge.py"
    linecache.checkcache(str(path))  # drop stale source lines if the file changed
    spec = importlib.util.spec_from_file_location(
        f"_judge_pin_{hashlib.sha256(str(path).encode()).hexdigest()[:12]}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return _sha256_text(inspect.getsource(mod.build_judge_prompt))


def pins_path(root: Path | None = None) -> Path:
    root = root or ROOT
    cfg = _load_config(root)
    return root / cfg["global"].get("pins", "bench/pins.json")


def compute_pins(root: Path | None = None, allow_missing: bool = False) -> dict:
    """Hash the fixed-benchmark surface. Reading tasks/*/{dev,test}.json only.

    allow_missing=True skips absent files instead of raising — for verifying a
    partial working copy (agent workspaces are seeded WITHOUT tasks/*/test.json).
    freeze always uses the strict form: a partial tree cannot define the benchmark.
    """
    root = root or ROOT
    cfg = _load_config(root)
    files: dict[str, str] = {}
    corpora: dict[str, str] = {}
    missing: list[str] = []
    for rel in HARNESS_FILES:
        p = root / rel
        if p.exists():
            files[rel] = _sha256_file(p)
        else:
            missing.append(rel)
    for task, tcfg in cfg["tasks"].items():
        # the config is always pinned; a variant task (extends: parent)
        # carries nothing else of its own — the asset files it inherits are
        # pinned under the parent task already.
        crel = f"task_configs/{task}.yaml"
        cp = root / crel
        if cp.exists():
            files[crel] = _sha256_file(cp)
        else:
            missing.append(crel)
        names = TASK_FILES if not tcfg.get("extends") else []
        for name in names:
            rel = f"workspace_setup/tasks/{task}/{name}"
            p = root / rel
            if p.exists():
                files[rel] = _sha256_file(p)
            else:
                missing.append(rel)
        # instructions: a block-path mapping (or a bare data_access path) —
        # any file a task stitches its AGENTS.md from is pinned too
        instr = tcfg.get("instructions")
        if isinstance(instr, str):
            instr = {"data_access": instr}
        for rel in sorted(set((instr or {}).values())):
            ip = root / rel
            if ip.exists():
                files[str(rel)] = _sha256_file(ip)
            else:
                missing.append(str(rel))
        for key in TASK_DATA_KEYS:
            rel = tcfg.get(key)
            if not rel:
                continue
            p = root / rel
            if p.exists():
                files[rel] = _sha256_file(p)
            else:
                missing.append(rel)
        crel = tcfg.get(TASK_CORPUS_KEY)
        if crel:
            cdir = root / crel
            # A corpus may legitimately be absent from a partial working copy
            # (it is huge); record the sentinel so freeze/verify stay symmetric.
            corpora[crel] = _sha256_tree(cdir) if cdir.is_dir() else "absent"
    if missing and not allow_missing:
        raise FileNotFoundError(f"cannot pin, missing files: {missing}")
    return {
        "version": PINS_VERSION,
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "eval_budget": cfg["global"]["eval_budget"],
        "judge_prompt_sha": judge_prompt_sha(root),
        "files": files,
        "corpora": corpora,
    }


def write_pins(root: Path | None = None) -> Path:
    root = root or ROOT
    pins = compute_pins(root)
    out = pins_path(root)
    out.write_text(json.dumps(pins, indent=2) + "\n")
    return out


def read_pins(root: Path | None = None) -> dict | None:
    p = pins_path(root or ROOT)
    if not p.exists():
        return None
    return json.loads(p.read_text())


def verify_pins(root: Path | None = None) -> list[str]:
    """Compare the current surface against bench/pins.json.

    Returns [] when clean; otherwise a list of human-readable mismatches.
    A missing pins file is itself a mismatch (the benchmark was never frozen).
    """
    root = root or ROOT
    pinned = read_pins(root)
    if pinned is None:
        return [f"no pins file at {pins_path(root)} — run `python bench.py freeze` first"]
    mismatches: list[str] = []
    current = compute_pins(root, allow_missing=True)
    if pinned.get("eval_budget") != current["eval_budget"]:
        mismatches.append(
            f"eval_budget changed: pinned {pinned.get('eval_budget')} != current {current['eval_budget']}")
    if pinned.get("judge_prompt_sha") != current["judge_prompt_sha"]:
        mismatches.append("judge prompt template (judge.build_judge_prompt source) changed")
    pf, cf = pinned.get("files", {}), current["files"]
    for rel in sorted(set(pf) | set(cf)):
        if rel not in pf:
            mismatches.append(f"{rel}: not in pins (new file — re-freeze deliberately)")
        elif rel not in cf:
            # A pinned file may legitimately be absent from a working copy (agent
            # workspaces never contain the held-out test.json). Absence cannot be
            # silent drift: anything that reads the file fails loudly without it.
            continue
        elif pf[rel] != cf[rel]:
            mismatches.append(f"{rel}: sha256 mismatch (file changed since freeze)")
    pc, cc = pinned.get("corpora", {}), current.get("corpora", {})
    for rel in sorted(set(pc) | set(cc)):
        cur = cc.get(rel, "absent")
        if cur == "absent":
            # Corpus dir not present in this tree (partial copy): content cannot be
            # checked here, and eval cannot run without it, so drift is not silent.
            continue
        pin = pc.get(rel)
        if pin is None:
            mismatches.append(f"{rel}: corpus not in pins (re-freeze deliberately)")
        elif pin != cur:
            mismatches.append(f"{rel}: corpus content drift (tree hash changed since freeze)")
    return mismatches


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Learning Agent integrity lock")
    ap.add_argument("cmd", choices=["freeze", "verify"])
    args = ap.parse_args()
    if args.cmd == "freeze":
        out = write_pins()
        print(f"[integrity] pinned {len(json.loads(out.read_text())['files'])} files -> {out}")
    else:
        problems = verify_pins()
        if not problems:
            print("[integrity] OK — benchmark surface matches bench/pins.json")
        else:
            print("[integrity] MISMATCH:")
            for m in problems:
                print(f"  - {m}")
            raise SystemExit(1)
