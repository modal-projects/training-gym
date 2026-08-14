"""Seed-tool registry + pure command classification for the observatory's
"learning timeline" (Learning Agent's thesis is "learning as tools": the agent generates
training data, trains, evaluates, and runs evolve recipes as ordinary shell
commands recorded in the trace).

Pure stdlib. Must NOT import from toolbox/ (or anything outside the standard
library) — the observatory ingests runs on machines that don't have the
repo's training/eval dependencies installed. The anti-drift guarantee that
this registry never silently misses a new toolbox/data_toolbox/gen/*.py
generator comes from a TEST (observatory/tests/test_learning.py) that walks
the repo checkout's gen/ directory; this module itself never touches the
filesystem.

Agent trace commands arrive as full shell strings, often chained with `&&`,
tolerant of arbitrary prefixes (cd, env vars, python3/python, uv run,
modal run, bash -c quoting), e.g.:

    cd ws && python3 toolbox/data_toolbox/gen/paraphrase.py --corpus c --out o
    python3 bench.py train --task dspy --rows data/x.jsonl --tag t1
    modal run pipeline/rl.py::rl_entry --task dspy --tag t1 --num-rollout 24

`classify_command` finds every seed-tool invocation in one such string (one
LearningAction-shaped dict per hit, in order of appearance) and does
best-effort flag extraction. `extract_script_paths` returns every other
explicit script token (for Task 6's invented-tool detection).
"""

from __future__ import annotations

import re

# ---- the seed-tool registry ------------------------------------------------
#
# Ordered (kind, tool, regex) entries, matched as command SUBSTRINGS (never by
# token position — commands may be prefixed with `cd x &&`, env vars, `python3`
# / `python`, `uv run`, `modal run`, `bash -c '...'`, etc.). `tool` is None for
# the one templated entry (data-gen scripts): its tool name is the captured
# stem, which is what makes any toolbox/data_toolbox/gen/<name>.py generator
# register automatically with no hardcoded list to keep in sync — the drift
# test in observatory/tests/test_learning.py guards this by enumerating the
# real gen/ directory in the repo checkout.
#
# `bench.py train` / `bench.py rl` / `bench.py score` are matched as a
# two-token PHRASE (allowing a run of whitespace between the tokens), never by
# the bare substring "bench.py" — a single `bench.py train` invocation must
# yield exactly one action, not also match `pipeline/train.py`'s pattern (it
# won't: the two patterns are disjoint substrings). `bench.py eval` / `judge`
# / `freeze` / `verify` / `leaderboard` intentionally match nothing here —
# they aren't one of the four learning kinds.
_DATA_GEN_RE = re.compile(r"toolbox/data_toolbox/gen/([A-Za-z0-9_\-]+)\.py")

# Folder-form tools: toolbox/<category>/…/<tool>/run.py — the current form
# for multi-file tools (dpo, opd, context_distill) and the 2026-08-03..05
# form for every tool; old traces keep classifying. The captured folder name
# is the tool identity; kind comes from the category.
def _folder_re(category: str) -> re.Pattern:
    return re.compile(
        rf"toolbox/{category}/(?:[A-Za-z0-9_\-]+/)*([A-Za-z0-9_\-]+)/run\.py")


# Single-file tools: toolbox/<category>/…/<name>.py — the current normal form
# (2026-08-05 layout). Two exclusions keep the pattern honest: path segments
# that are cloned reference packages (repos.yaml dests — their internal
# scripts are not Learning Agent tools; _REPO_DIRS mirrors repos.yaml, drift-tested in
# observatory/tests/test_learning.py), and filenames that are entrypoints/
# internals/libraries rather than tools (run.py, _-prefixed, registry,
# corpus_sampling).
_REPO_DIRS = ("axolotl", "training_gym",
              # dropped from repos.yaml (2026-08-12/13 toolbox slimming) —
              # keep excluding forever so old run records still classify
              "self_distillation", "cartridges", "gepa", "unsloth", "trl",
              "slime", "miles")
_SEG = rf"(?:(?!(?:{'|'.join(_REPO_DIRS)})/)[A-Za-z0-9_\-]+/)*"
_NOT_TOOL_FILE = r"(?!run\.py)(?!_)(?!registry\.py)(?!corpus_sampling\.py)(?!__init__)"


def _file_re(category: str) -> re.Pattern:
    return re.compile(
        rf"toolbox/{category}/{_SEG}{_NOT_TOOL_FILE}([A-Za-z0-9_\-]+)\.py")


# Catch-alls for tools in categories this registry doesn't know (any
# toolbox/*_tool/ directory is a category, and the set may grow mid-run).
# Kind "tool" is the honest default; the collector refines it
# from the tool's own card when the workspace snapshot carries one.
_FOLDER_CATCHALL_RE = re.compile(
    r"toolbox/[A-Za-z0-9_\-]+/(?:[A-Za-z0-9_\-]+/)*([A-Za-z0-9_\-]+)/run\.py")
_FILE_CATCHALL_RE = re.compile(
    rf"toolbox/[A-Za-z0-9_\-]+/{_SEG}{_NOT_TOOL_FILE}([A-Za-z0-9_\-]+)\.py")

_REGISTRY: list[tuple[str, "str | None", re.Pattern]] = [
    ("data", None, _DATA_GEN_RE),
    # The category taxonomy (mirrors CATEGORY_KINDS in
    # observatory/validate_tools.py; this module stays filesystem-free by
    # design). Both forms per category: single-file (current) and
    # folder/run.py (multi-file tools + 2026-08 traces):
    ("data", None, _file_re("data_tool")),
    ("train", None, _file_re("training_tool")),
    ("eval", None, _file_re("eval_tool")),
    ("harness", None, _file_re("harness_tool")),
    ("evolve", None, _file_re("self_evolve_tool")),
    ("infra", None, _file_re("inference_tool")),
    # support dirs that are seed tools too — without these they fall to the
    # catch-all kind "tool" (which the dashboard used to mis-badge "invented")
    ("infra", None, _file_re("gpu_tools")),
    ("infra", None, _file_re("api_clients")),
    # agents write their OWN trainers and launch them through gpu_launcher
    # (observed 2026-08-13: /root/sft_train.py, sft_train_lora.py, merge_lora)
    # — a launcher-only match reads as "infra" and training vanishes from the
    # learning timeline. Payloads with training markers classify as train.
    ("train", None, re.compile(
        r"gpu_launcher\.py[^|;&]*\b(\w*(?:sft|lora|dpo|grpo|finetun|distill)\w*\.py)")),
    ("data", None, _folder_re("data_tool")),
    ("train", None, _folder_re("training_tool")),
    ("eval", None, _folder_re("eval_tool")),
    ("harness", None, _folder_re("harness_tool")),
    ("evolve", None, _folder_re("self_evolve_tool")),
    ("infra", None, _folder_re("inference_tool")),
    ("harness", None, _folder_re("agentic_toolbox")),
    # Legacy category dirs — old traces and the single-file implementation
    # paths keep classifying forever:
    ("data", None, _folder_re("data_toolbox")),
    ("eval", None, _folder_re("eval_toolbox")),
    ("eval", None, _folder_re("harness_toolbox")),
    ("train", None, _folder_re("training_toolbox")),
    ("evolve", None, _folder_re("evolve")),
    ("train", "sft", re.compile(r"bench\.py\s+train\b")),
    ("train", "sft", re.compile(r"pipeline/train\.py")),
    ("train", "rl", re.compile(r"bench\.py\s+rl\b")),
    ("train", "rl", re.compile(r"pipeline/rl\.py")),
    ("eval", "rubric_eval", re.compile(r"toolbox/eval_toolbox/rubric_eval\.py")),
    ("eval", "bench_score", re.compile(r"bench\.py\s+score\b")),
    ("evolve", "run_recipe", re.compile(r"toolbox/evolve/run_recipe\.py")),
]

# Flags surfaced into LearningAction.args when present after a match
# (best-effort; both `--flag value` and `--flag=value` forms). Dict keys use
# argparse's own dest convention (hyphens -> underscores).
_FLAGS = ("method", "tag", "rows", "budget", "num-rollout", "dev")

# A chained shell command's flags belong to the sub-command they trail, not
# the next one in the chain — stop flag-scanning at the next boundary.
_BOUNDARY_RE = re.compile(r"&&|\||;")
_FLAG_VALUE_RE = re.compile(r'"[^"]*"|\'[^\']*\'|\S+')


def _registry_matches(cmd: str) -> list[tuple[int, int, str, str]]:
    """(start, end, kind, tool) for every registry hit in `cmd`, sorted by
    position of appearance. A set() dedups the (vanishingly unlikely) case of
    two patterns matching the identical span. The folder catch-all runs last
    and only claims spans no category-specific pattern claimed."""
    hits = set()
    for kind, tool, rx in _REGISTRY:
        for m in rx.finditer(cmd):
            hits.add((m.start(), m.end(), kind, tool if tool is not None else m.group(1)))
    claimed = {(s, e) for s, e, _, _ in hits}
    for rx in (_FOLDER_CATCHALL_RE, _FILE_CATCHALL_RE):
        for m in rx.finditer(cmd):
            if (m.start(), m.end()) not in claimed:
                hits.add((m.start(), m.end(), "tool", m.group(1)))
                claimed.add((m.start(), m.end()))
    return sorted(hits, key=lambda h: h[0])


def _segment_after(cmd: str, start: int) -> str:
    """Text from `start` to the next shell chain boundary (&&, |, ;), or end
    of string."""
    m = _BOUNDARY_RE.search(cmd, start)
    return cmd[start:m.start()] if m else cmd[start:]


def _strip_quotes(tok: str) -> str:
    if len(tok) >= 2 and tok[0] == tok[-1] and tok[0] in ("'", '"'):
        return tok[1:-1]
    return tok


def _extract_flags(segment: str) -> dict:
    args: dict = {}
    for flag in _FLAGS:
        m = re.search(rf"--{re.escape(flag)}(?:=|\s+)", segment)
        if not m:
            continue
        vm = _FLAG_VALUE_RE.match(segment, m.end())
        if not vm:
            continue
        args[flag.replace("-", "_")] = _strip_quotes(vm.group(0))
    return args


def classify_command(cmd: str) -> list[dict]:
    """Every seed-tool invocation found in `cmd`, in order of appearance (a
    `&&`/`;`/`|`-chained command may hit several; one action per hit).

    Each returned dict carries the LearningAction (see schema.py) fields this
    pure string parser can know on its own: `kind`, `tool`,
    `provenance="seed"` (a registry match is by definition a seed tool),
    `command` (the full input string, for display), and `args` (best-effort
    surfaced flags). The collector adds `event_i`/`ts`/`nth_use` once it knows
    which trace event this command came from and how it orders against the
    run's other actions.
    """
    if not cmd:
        return []
    actions = []
    for start, end, kind, tool in _registry_matches(cmd):
        actions.append({
            "kind": kind,
            "tool": tool,
            "provenance": "seed",
            "command": cmd,
            "args": _extract_flags(_segment_after(cmd, end)),
        })
    return actions


_TOKEN_RE = re.compile(r"\S+")


def extract_script_paths(cmd: str) -> list[str]:
    """Explicit .py/.sh script tokens in `cmd`, in order of appearance,
    deduped, excluding tokens that overlap a registry match (those are
    already accounted for by `classify_command`). Common quoting is stripped.

    For Task 6's invented-tool detection: a token that survives this filter
    and isn't in the run's seed manifest is a candidate invented tool. Only
    tokens *ending in* .py/.sh are surfaced (not every `/`-containing token,
    e.g. a --corpus/--rows data path) — those aren't scripts, and counting
    them would pollute the invented-tool count with ordinary data/output
    paths.
    """
    if not cmd:
        return []
    matches = _registry_matches(cmd)
    seen: set[str] = set()
    out: list[str] = []
    repo_seg = re.compile(rf"toolbox/.*\b(?:{'|'.join(_REPO_DIRS)})/")
    for tm in _TOKEN_RE.finditer(cmd):
        tok = _strip_quotes(tm.group(0))
        if not (tok.endswith(".py") or tok.endswith(".sh")):
            continue
        if repo_seg.search(tok):
            continue  # cloned-package internals are not invented tools
        if any(tm.start() < end and start < tm.end() for start, end, _, _ in matches):
            continue
        if tok in seen:
            continue
        seen.add(tok)
        out.append(tok)
    return out
