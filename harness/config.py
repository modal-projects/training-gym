"""Learning Agent config resolution — one loader for the whole bench.

Layout on disk (the spec, all pinned by bench/pins.json):

    bench/config.yaml        GLOBAL pins only: base model, judge, volumes, splits.
    task_configs/<T>.yaml    THE per-task config. A task exists iff this file
                             exists — bench.py choices, runner whitelists, and
                             integrity pinning all derive from it. Holds identity
                             (archetype, corpus, dev/test/sys, secondary) plus
                             per-task defaults (eval/env/agent/judge/session).
    tasks/<T>/               the task ASSETS only (corpus, dev/test json,
                             sys.txt, task.md, brief.md) — the folder seeding
                             copies into a workspace. The config never enters
                             a workspace: the operator reads it, agents don't.

Resolution for a run (later wins):

    task_configs/<T>.yaml  ->  --config <override.yaml>  ->  CLI flags

`resolve()` performs the first two; the CLI layer applies its own flags on top
(a flag the user didn't pass falls back to the resolved value). The RESOLVED
config is what a run snapshots to runs/<tag>/run_config.yaml — one file that
reproduces the run.

`load_config()` returns the legacy combined view {"global": …, "tasks": {…}} so
bench.py / judge_cli.py / integrity.py keep their access patterns unchanged.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import yaml

GLOBAL_CONFIG = "bench/config.yaml"
TASK_CONFIG_DIR = "task_configs"


def task_config_path(root: Path, task: str) -> Path:
    return Path(root) / TASK_CONFIG_DIR / f"{task}.yaml"

# Keys a task.yaml must carry, by archetype. `validate_task` reports what's
# missing; tests/test_config.py enforces it repo-wide so a half-registered task
# is a test failure, not a runtime surprise.
REQUIRED_COMMON = ("task", "archetype", "dev", "test", "sys")
REQUIRED_BY_ARCHETYPE = {
    "qa": ("corpus", "glob", "secondary"),
    "agentic": ("adapter", "env"),
}


def load_global(root: Path) -> dict:
    """The `global:` section of bench/config.yaml."""
    cfg = yaml.safe_load((Path(root) / GLOBAL_CONFIG).read_text())
    return cfg["global"]


def known_tasks(root: Path) -> list[str]:
    """A task exists iff task_configs/<T>.yaml exists. Sorted for stable CLIs."""
    tdir = Path(root) / TASK_CONFIG_DIR
    if not tdir.is_dir():
        return []
    return sorted(p.stem for p in tdir.glob("*.yaml"))


def load_task(root: Path, task: str) -> dict:
    """Parse task_configs/<task>.yaml. The `task:` field must match the file
    stem — a copy-paste of another task's config must fail loudly, not
    mis-score.

    A task may declare `extends: <parent>`: the parent's task.yaml is loaded
    first and the child's keys are deep-merged over it (the mechanism behind
    variant tasks like fav2_rl — same corpus/dev/test by inheritance, its own
    toolbox and instructions). One level only: a parent may not itself extend.
    The merged config keeps the CHILD's `task:` identity."""
    p = task_config_path(root, task)
    if not p.exists():
        raise SystemExit(f"[config] unknown task {task!r}: no {p}")
    tcfg = yaml.safe_load(p.read_text())
    if not isinstance(tcfg, dict):
        raise SystemExit(f"[config] {p} is not a mapping")
    if tcfg.get("task") != task:
        raise SystemExit(f"[config] {p}: task: {tcfg.get('task')!r} != dir {task!r}")
    parent = tcfg.pop("extends", None)
    if parent:
        pp = task_config_path(root, str(parent))
        if not pp.exists():
            raise SystemExit(f"[config] {p}: extends {parent!r} but no {pp}")
        pcfg = yaml.safe_load(pp.read_text())
        if not isinstance(pcfg, dict):
            raise SystemExit(f"[config] {pp} is not a mapping")
        if pcfg.get("extends"):
            raise SystemExit(f"[config] {pp}: chained extends is not allowed")
        tcfg = deep_merge(pcfg, tcfg)
        tcfg["task"] = task
        tcfg["extends"] = str(parent)   # kept for asset resolution (corpus paths)
    tcfg.setdefault("archetype", "qa")
    return tcfg


def load_config(root: Path) -> dict:
    """Legacy combined view: {"global": …, "tasks": {name: task_cfg}}."""
    return {"global": load_global(root),
            "tasks": {t: load_task(root, t) for t in known_tasks(root)}}


def deep_merge(base: dict, over: dict) -> dict:
    """Recursive dict merge, `over` wins; lists/scalars replace wholesale."""
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def resolve(root: Path, task: str, override_path: str | Path | None = None) -> dict:
    """task.yaml defaults <- optional override YAML. Returns the merged task-level
    config. The override file uses the SAME schema as task.yaml (any subset of
    keys); if it names a task, it must be this one."""
    tcfg = load_task(root, task)
    if not override_path:
        return tcfg
    op = Path(override_path)
    if not op.exists():
        raise SystemExit(f"[config] override not found: {op}")
    over = yaml.safe_load(op.read_text()) or {}
    if not isinstance(over, dict):
        raise SystemExit(f"[config] override {op} is not a mapping")
    if over.get("task") not in (None, task):
        raise SystemExit(f"[config] override {op} is for task {over.get('task')!r}, "
                         f"not {task!r}")
    return deep_merge(tcfg, over)


def validate_task(tcfg: dict) -> list[str]:
    """Human-readable list of missing/invalid required keys ([] == valid)."""
    problems = []
    for k in REQUIRED_COMMON:
        if not tcfg.get(k):
            problems.append(f"missing required key: {k}")
    arch = tcfg.get("archetype", "qa")
    if arch not in REQUIRED_BY_ARCHETYPE:
        problems.append(f"unknown archetype: {arch!r}")
        return problems
    for k in REQUIRED_BY_ARCHETYPE[arch]:
        if tcfg.get(k) in (None, ""):
            problems.append(f"[{arch}] missing required key: {k}")
    return problems


def config_sha(root: Path, task: str) -> str:
    """Provenance hash of the config that governed a run: global config bytes +
    this task's config bytes (path-labelled so file swaps can't collide)."""
    h = hashlib.sha256()
    for rel in (GLOBAL_CONFIG, f"{TASK_CONFIG_DIR}/{task}.yaml"):
        h.update(rel.encode())
        h.update(b"\0")
        h.update((Path(root) / rel).read_bytes())
        h.update(b"\n")
    return h.hexdigest()


def snapshot(resolved: dict, dest: Path) -> Path:
    """Write the resolved run config next to the run's artifacts — the single
    file that reproduces the run. YAML for humans; key order preserved."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(yaml.safe_dump(resolved, sort_keys=False))
    return dest




# ---- toolbox composition (what gets seeded into a workspace) -----------------

HARNESSES_BY_ARCHETYPE = {
    "qa": ["react_loop", "completion_qa"],
    "agentic": ["react_env_agent", "react_tool_agent", "mini_swe_agent"],
}
ALL_TRAINING_METHODS = ["sft", "dpo", "rl"]     # the default set; tasks opt into more
# method -> pinned package (toolbox_bank/repos.yaml key) that trains it
PACKAGES_BY_METHOD = {"sft": "axolotl", "dpo": "axolotl", "rl": "training_gym",
                      "opd": "training_gym", "sdft": "self_distillation"}
# method -> data-card families copied in from toolbox_bank/data_tool/
CARDS_BY_METHOD = {"sft": ["synthetic", "agentic"], "dpo": ["preference"],
                   "rl": ["rl"], "opd": ["rl"], "sdft": ["self_distill"]}


def instructions_config(tcfg: dict) -> dict:
    """Which md files fill the AGENTS.md template slots for this task.

    `instructions:` in task.yaml is a mapping with optional keys objective /
    data_access / setup; a bare string is shorthand for the data_access
    override (the historical form). Defaults: objective follows the task's
    archetype; data_access follows the launch track (resolved at seed time,
    so it is None here when not overridden); setup follows
    global.setup_instructions (also resolved by the consumer)."""
    ins = tcfg.get("instructions")
    if isinstance(ins, str):
        ins = {"data_access": ins}
    ins = ins or {}
    if not isinstance(ins, dict):
        raise SystemExit("[config] instructions: must be a mapping or a path")
    unknown = set(ins) - {"objective", "data_access", "setup", "rules", "harness", "tips"}
    if unknown:
        raise SystemExit(f"[config] instructions: unknown keys {sorted(unknown)}")
    archetype = tcfg.get("archetype", "qa")
    return {
        "objective": ins.get("objective") or f"instructions/objective/{archetype}.md",
        "data_access": ins.get("data_access"),
        "setup": ins.get("setup"),
        "rules": ins.get("rules"),
        "harness": ins.get("harness"),
        "tips": ins.get("tips"),
    }


def toolbox_config(tcfg: dict) -> dict:
    """The resolved `toolbox:` block for a task config, with defaults.

    Returns {"harnesses": [...], "training": [...], "packages": [...],
    "seed_dev": bool, "seed_corpus": bool, "instructions": {...} (see
    instructions_config), "assets_task": str}. `assets_task` is where
    gitignored inputs (corpus, dev.json) live on disk: the parent task for
    variants, the task itself otherwise."""
    tb = tcfg.get("toolbox") or {}
    if not isinstance(tb, dict):
        raise SystemExit("[config] toolbox: must be a mapping")
    archetype = tcfg.get("archetype", "qa")
    harnesses = list(tb.get("harnesses") or HARNESSES_BY_ARCHETYPE[archetype])
    training = list(tb.get("training") or ALL_TRAINING_METHODS)
    for m in training:
        if m not in PACKAGES_BY_METHOD:
            raise SystemExit(f"[config] unknown training method {m!r} "
                             f"(known: {ALL_TRAINING_METHODS})")
    packages = sorted({PACKAGES_BY_METHOD[m] for m in training})
    cards = sorted({c for m in training for c in CARDS_BY_METHOD[m]})
    seed = tcfg.get("seed") or {}
    return {
        "harnesses": harnesses,
        "training": training,
        "packages": packages,
        "cards": cards,
        "seed_dev": bool(seed.get("dev", True)),
        "seed_corpus": bool(seed.get("corpus", True)),
        "instructions": instructions_config(tcfg),
        "assets_task": tcfg.get("extends") or tcfg["task"],
    }


def _emit_seed_env(root: Path, task: str) -> None:
    """Print shell-evalable lines describing the seeding plan for one task —
    the bridge between this loader and workspace_setup/prepare_workspace.sh."""
    tcfg = load_task(root, task)
    tb = toolbox_config(tcfg)
    print(f'LEARNING_AGENT_TB_HARNESSES="{" ".join(tb["harnesses"])}"')
    print(f'LEARNING_AGENT_TB_TRAINING="{" ".join(tb["training"])}"')
    print(f'LEARNING_AGENT_TB_PACKAGES="{" ".join(tb["packages"])}"')
    print(f'LEARNING_AGENT_TB_CARDS="{" ".join(tb["cards"])}"')
    all_cards = sorted({c for cs in CARDS_BY_METHOD.values() for c in cs})
    print(f'LEARNING_AGENT_TB_ALL_CARDS="{" ".join(all_cards)}"')   # every selectable family (for pruning)
    print(f'LEARNING_AGENT_TB_SEED_DEV={"1" if tb["seed_dev"] else "0"}')
    print(f'LEARNING_AGENT_TB_SEED_CORPUS={"1" if tb["seed_corpus"] else "0"}')
    ins = tb["instructions"]
    print(f'LEARNING_AGENT_TB_INSTR_OBJECTIVE="{ins["objective"]}"')
    print(f'LEARNING_AGENT_TB_INSTR_DATA_ACCESS="{ins["data_access"] or ""}"')   # empty -> track default
    print(f'LEARNING_AGENT_TB_INSTR_SETUP="{ins["setup"] or ""}"')               # empty -> global default
    print(f'LEARNING_AGENT_TB_INSTR_RULES="{ins["rules"] or ""}"')               # empty -> rules/default.md
    print(f'LEARNING_AGENT_TB_INSTR_HARNESS="{ins["harness"] or ""}"')           # empty -> harness/<archetype>.md
    print(f'LEARNING_AGENT_TB_INSTR_TIPS="{ins["tips"] or ""}"')                 # empty -> tips/<archetype>.md
    print(f'LEARNING_AGENT_TB_ASSETS_TASK="{tb["assets_task"]}"')
    print(f'LEARNING_AGENT_TB_ARCHETYPE="{tcfg.get("archetype", "qa")}"')


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Learning Agent config resolution")
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--task", default="")
    ap.add_argument("--config", default="", help="override YAML")
    ap.add_argument("--emit-seed-env", action="store_true",
                    help="print shell-evalable seeding plan for the task")
    args = ap.parse_args()
    root = Path(args.root)
    if args.emit_seed_env:
        if not args.task:
            raise SystemExit("--emit-seed-env requires --task")
        _emit_seed_env(root, args.task)
    elif not args.task:
        print("\n".join(known_tasks(root)))
    else:
        rcfg = resolve(root, args.task, args.config or None)
        problems = validate_task(rcfg)
        print(json.dumps(rcfg, indent=2))
        if problems:
            raise SystemExit("[config] INVALID: " + "; ".join(problems))
